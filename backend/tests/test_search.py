from tests.conftest import embedding, make_document
from think9.models import ParsedChunk
from think9.retrieval.search import dense_search, sparse_search
from think9.store.repository import Repository


def _seed(conn, *, acl, text, vector, heading="Pricing"):
    repo = Repository(conn)
    doc = make_document(acl=acl)
    repo.upsert_document(doc)
    repo.insert_chunks(doc.id, [ParsedChunk(0, heading, text)], [vector])
    return doc


def test_dense_search_returns_ranked_candidates(conn):
    _seed(conn, acl=("procurement",), text="amber glass Rs 22.10", vector=embedding(0.9))
    _seed(conn, acl=("procurement",), text="annual leave policy", vector=embedding(-0.9))

    results = dense_search(conn, embedding(0.9), user_groups=["procurement"])

    assert results[0].text == "amber glass Rs 22.10"
    assert results[0].rank == 1
    assert results[0].source == "dense"


def test_dense_search_hides_chunks_the_user_cannot_open(conn):
    """Filtering happens in SQL, before generation. Filtering after generation leaks."""
    _seed(conn, acl=("legal",), text="settlement terms", vector=embedding(0.9))

    assert dense_search(conn, embedding(0.9), user_groups=["procurement"]) == []


def test_dense_search_admits_a_chunk_when_any_group_matches(conn):
    _seed(conn, acl=("procurement", "legal"), text="indemnity", vector=embedding(0.9))

    assert len(dense_search(conn, embedding(0.9), user_groups=["legal", "brand_ops"])) == 1


def test_a_user_with_no_groups_retrieves_nothing(conn):
    _seed(conn, acl=("procurement",), text="amber glass", vector=embedding(0.9))

    assert dense_search(conn, embedding(0.9), user_groups=[]) == []


def test_sparse_search_finds_an_exact_entity_token(conn):
    """Vendor names and SKU codes are exactly where embeddings blur and keywords win."""
    _seed(conn, acl=("procurement",), text="Korent Glassworks SKU AMB-50-FL", vector=embedding(0.1))
    _seed(conn, acl=("procurement",), text="general packaging notes", vector=embedding(0.1))

    results = sparse_search(conn, "AMB-50-FL", user_groups=["procurement"])

    assert len(results) == 1
    assert "AMB-50-FL" in results[0].text
    assert results[0].source == "sparse"


def test_sparse_search_also_enforces_acl(conn):
    _seed(conn, acl=("legal",), text="Korent indemnity clause 7.3", vector=embedding(0.1))

    assert sparse_search(conn, "indemnity clause", user_groups=["procurement"]) == []


def test_sparse_search_returns_nothing_for_an_absent_term(conn):
    _seed(conn, acl=("procurement",), text="amber glass pricing", vector=embedding(0.1))

    assert sparse_search(conn, "freight insurance excess", user_groups=["procurement"]) == []


def test_dense_and_sparse_rank_independently(conn):
    """The two arms disagree, which is the whole reason fusion exists."""
    _seed(conn, acl=("procurement",), text="Korent AMB-50-FL", vector=embedding(-0.9))
    _seed(conn, acl=("procurement",), text="amber glass jars", vector=embedding(0.9))

    dense = dense_search(conn, embedding(0.9), user_groups=["procurement"])
    sparse = sparse_search(conn, "AMB-50-FL", user_groups=["procurement"])

    assert dense[0].text == "amber glass jars"
    assert sparse[0].text == "Korent AMB-50-FL"


def test_limit_is_respected(conn):
    for i in range(5):
        _seed(conn, acl=("procurement",), text=f"amber glass {i}", vector=embedding(0.1 * i))

    assert len(dense_search(conn, embedding(0.2), user_groups=["procurement"], limit=3)) == 3
