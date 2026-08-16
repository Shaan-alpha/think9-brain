from datetime import date
from uuid import uuid4

import pytest

from tests.conftest import embedding, make_document
from think9.models import Owner, ParsedChunk
from think9.store.repository import Repository


def test_upsert_document_then_read_it_back(conn):
    repo = Repository(conn)
    doc = make_document()

    repo.upsert_document(doc)

    stored = repo.get_document(doc.id)
    assert stored is not None
    assert stored.title == "Korent Quote"
    assert stored.acl == ("procurement",)
    assert stored.effective_date == date(2026, 1, 5)
    assert stored.is_superseded is False


def test_upsert_is_idempotent_on_source_id(conn):
    repo = Repository(conn)
    doc = make_document(title="First")
    repo.upsert_document(doc)
    repo.upsert_document(make_document(id=doc.id, source_id=doc.source_id, title="Second"))

    assert repo.get_document(doc.id).title == "Second"
    assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 1


def test_get_documents_returns_every_match_keyed_by_id(conn):
    """One round trip for the whole shortlist.

    Enriching a shortlist one document at a time was eight sequential round trips per
    question to a database in another region, which is seconds of latency before a model
    has done anything.
    """
    repo = Repository(conn)
    docs = [make_document(source_id=f"file-{i}", title=f"Doc {i}") for i in range(3)]
    for doc in docs:
        repo.upsert_document(doc)

    found = repo.get_documents([d.id for d in docs])

    assert set(found) == {d.id for d in docs}
    assert sorted(d.title for d in found.values()) == ["Doc 0", "Doc 1", "Doc 2"]


def test_get_documents_omits_ids_that_are_not_there(conn):
    """A chunk whose document has been deleted must drop out, not raise."""
    repo = Repository(conn)
    doc = make_document()
    repo.upsert_document(doc)

    found = repo.get_documents([doc.id, uuid4()])

    assert list(found) == [doc.id]


def test_get_documents_asks_nothing_when_given_nothing(conn):
    """A refusal retrieves no chunks, and `WHERE id = ANY('{}')` is a wasted round trip."""
    assert Repository(conn).get_documents([]) == {}


def test_insert_chunks_stores_embeddings_and_generates_tsv(conn):
    repo = Repository(conn)
    doc = make_document()
    repo.upsert_document(doc)

    ids = repo.insert_chunks(
        doc.id,
        [ParsedChunk(ordinal=0, heading_path="Pricing", text="Rs 22.10 per unit")],
        [embedding(0.2)],
    )

    assert len(ids) == 1
    row = conn.execute("SELECT tsv IS NOT NULL FROM chunks WHERE id = %s", (ids[0],)).fetchone()
    assert row[0] is True


def test_reinserting_chunks_replaces_rather_than_duplicates(conn):
    repo = Repository(conn)
    doc = make_document()
    repo.upsert_document(doc)
    chunk = ParsedChunk(ordinal=0, heading_path="Pricing", text="Rs 22.10")

    repo.insert_chunks(doc.id, [chunk], [embedding(0.2)])
    repo.insert_chunks(doc.id, [chunk], [embedding(0.2)])

    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1


def test_chunks_and_embeddings_must_be_the_same_length(conn):
    repo = Repository(conn)
    doc = make_document()
    repo.upsert_document(doc)

    with pytest.raises(ValueError, match="same length"):
        repo.insert_chunks(doc.id, [ParsedChunk(0, "h", "t")], [])


def test_deleting_a_document_cascades_to_its_chunks(conn):
    repo = Repository(conn)
    doc = make_document()
    repo.upsert_document(doc)
    repo.insert_chunks(doc.id, [ParsedChunk(0, "h", "t")], [embedding()])

    conn.execute("DELETE FROM documents WHERE id = %s", (doc.id,))
    conn.commit()

    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0


def test_find_owner_resolves_brand_and_function(conn):
    repo = Repository(conn)
    repo.upsert_owner(Owner("nuvia", "procurement", "Priya Nair", "priya@think9.test"))

    owner = repo.find_owner("nuvia", "procurement")

    assert owner is not None
    assert owner.person_name == "Priya Nair"
    assert repo.find_owner("grove", "procurement") is None


def test_upsert_owner_is_idempotent_on_brand_and_function(conn):
    repo = Repository(conn)
    repo.upsert_owner(Owner("nuvia", "procurement", "Priya Nair", "priya@think9.test"))
    repo.upsert_owner(Owner("nuvia", "procurement", "Arun Menon", "arun@think9.test"))

    assert repo.find_owner("nuvia", "procurement").person_name == "Arun Menon"
    assert conn.execute("SELECT count(*) FROM owners").fetchone()[0] == 1


def test_mark_superseded_sets_the_flag_on_the_displaced_document(conn):
    repo = Repository(conn)
    old = make_document(title="Korent 2024", effective_date=date(2024, 3, 12))
    repo.upsert_document(old)
    repo.upsert_document(
        make_document(title="Korent 2026", effective_date=date(2026, 1, 8), supersedes_id=old.id)
    )

    repo.mark_superseded()

    assert repo.get_document(old.id).is_superseded is True


def test_mark_superseded_leaves_current_documents_alone(conn):
    repo = Repository(conn)
    doc = make_document()
    repo.upsert_document(doc)

    repo.mark_superseded()

    assert repo.get_document(doc.id).is_superseded is False
