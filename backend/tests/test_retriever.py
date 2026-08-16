from datetime import date

import pytest

from tests.conftest import make_document
from think9.models import ParsedChunk
from think9.retrieval.embed import Embedder
from think9.retrieval.rerank import Reranker
from think9.retrieval.retriever import Retriever
from think9.store.repository import Repository


@pytest.fixture(scope="session")
def embedder():
    return Embedder()


@pytest.fixture(scope="session")
def reranker():
    return Reranker()


def _seed(conn, embedder, doc, text, heading="Pricing"):
    repo = Repository(conn)
    repo.upsert_document(doc)
    chunk = ParsedChunk(0, heading, text)
    repo.insert_chunks(doc.id, [chunk], embedder.embed_chunks([chunk]))
    return doc


def test_retrieval_returns_chunks_with_coverage_and_an_as_of_date(conn, embedder, reranker):
    _seed(conn, embedder, make_document(), "50ml amber glass is Rs 22.10 per unit")
    retriever = Retriever(conn, embedder, reranker)

    result = retriever.retrieve("what do we pay for amber glass", "factual_lookup", ["procurement"])

    assert result.chunks
    assert result.coverage != 0.0
    assert result.as_of == date(2026, 1, 5)


def test_enrichment_takes_one_query_for_the_whole_shortlist(conn, embedder, reranker, monkeypatch):
    """Guards against the N+1 returning.

    Enrichment used to fetch one document per shortlisted chunk — up to eight sequential
    round trips per question, to a database the deployed API reaches across the Pacific.
    Nothing on the retrieval path may fetch documents singly any more.
    """
    for i in range(4):
        _seed(
            conn,
            embedder,
            make_document(source_id=f"file-{i}", title=f"Quote {i}"),
            f"50ml amber glass is Rs 22.1{i} per unit",
        )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("retrieval must not fetch documents one at a time")

    monkeypatch.setattr(Repository, "get_document", forbidden)

    result = Retriever(conn, embedder, reranker).retrieve(
        "what do we pay for amber glass", "factual_lookup", ["procurement"]
    )

    assert len(result.chunks) >= 2
    assert all(chunk.document is not None for chunk in result.chunks)


def test_a_chunk_whose_document_has_gone_is_dropped(conn, embedder, reranker):
    """It used to fall out via a None return; batching must not turn that into a KeyError."""
    doc = _seed(conn, embedder, make_document(), "50ml amber glass is Rs 22.10 per unit")
    _seed(conn, embedder, make_document(source_id="file-2", title="Other"), "amber glass pricing")
    conn.execute("DELETE FROM documents WHERE id = %s", (doc.id,))
    conn.commit()

    result = Retriever(conn, embedder, reranker).retrieve(
        "what do we pay for amber glass", "factual_lookup", ["procurement"]
    )

    assert all(chunk.document.id != doc.id for chunk in result.chunks)


def test_trace_records_every_stage(conn, embedder, reranker):
    _seed(conn, embedder, make_document(), "50ml amber glass is Rs 22.10 per unit")
    retriever = Retriever(conn, embedder, reranker)

    trace = retriever.retrieve("amber glass", "factual_lookup", ["procurement"]).trace

    assert set(trace) >= {"dense", "sparse", "fused", "reranked", "demoted"}


def test_disabling_hybrid_skips_the_sparse_arm(conn, embedder, reranker):
    _seed(conn, embedder, make_document(), "50ml amber glass is Rs 22.10 per unit")
    retriever = Retriever(conn, embedder, reranker)

    trace = retriever.retrieve(
        "amber glass", "factual_lookup", ["procurement"], use_hybrid=False
    ).trace

    assert trace["sparse"] == []


def test_nothing_retrievable_means_zero_coverage(conn, embedder, reranker):
    retriever = Retriever(conn, embedder, reranker)

    result = retriever.retrieve("freight insurance excess", "factual_lookup", ["procurement"])

    assert result.chunks == []
    assert result.coverage == 0.0
    assert result.as_of is None


def test_acl_is_enforced_through_the_whole_pipeline(conn, embedder, reranker):
    _seed(conn, embedder, make_document(acl=("legal",)), "settlement terms are confidential")
    retriever = Retriever(conn, embedder, reranker)

    result = retriever.retrieve("settlement terms", "factual_lookup", ["procurement"])

    assert result.chunks == []


def test_a_superseded_chunk_is_demoted_and_recorded_in_the_trace(conn, embedder, reranker):
    """The end-to-end temporal case: the dead price must not lead, and the trace must say why."""
    old = make_document(title="Korent 2024", effective_date=date(2024, 3, 12))
    _seed(conn, embedder, old, "50ml amber glass jar is Rs 18.40 per unit")
    _seed(
        conn,
        embedder,
        make_document(title="Korent 2026", effective_date=date(2026, 1, 8), supersedes_id=old.id),
        "50ml amber glass jar is Rs 22.10 per unit",
    )
    Repository(conn).mark_superseded()
    retriever = Retriever(conn, embedder, reranker)

    result = retriever.retrieve(
        "what do we pay for 50ml amber glass", "factual_lookup", ["procurement"]
    )

    assert "22.10" in result.chunks[0].text
    assert result.chunks[0].demoted is False
    assert result.as_of == date(2026, 1, 8)
    assert any("Korent 2024" == d["title"] for d in result.trace["demoted"])


def test_decision_archaeology_keeps_the_superseded_document_available(conn, embedder, reranker):
    old = make_document(title="Panel 2025", effective_date=date(2025, 5, 14))
    _seed(conn, embedder, old, "The mango variant scored lowest on purchase intent")
    _seed(
        conn,
        embedder,
        make_document(title="Panel 2026", effective_date=date(2026, 2, 20), supersedes_id=old.id),
        "Fig and cedar led on purchase intent",
    )
    Repository(conn).mark_superseded()
    retriever = Retriever(conn, embedder, reranker)

    result = retriever.retrieve(
        "why did we discontinue the mango variant", "decision_archaeology", ["procurement"]
    )

    assert any("mango" in c.text for c in result.chunks)
    assert all(c.demoted is False for c in result.chunks)
