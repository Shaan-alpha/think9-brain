from uuid import uuid4

from think9.models import Candidate
from think9.retrieval.rerank import Reranker


def _c(text, rank):
    return Candidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        text=text,
        heading_path="h",
        score=0.5,
        rank=rank,
        source="fused",
    )


def test_reranker_promotes_the_genuinely_relevant_passage():
    candidates = [
        _c("Employees accrue 18 days of annual leave per year.", 1),
        _c("50ml amber glass jars are priced at Rs 22.10 per unit.", 2),
    ]

    reranked = Reranker().rerank("what do we pay for amber glass", candidates)

    assert "Rs 22.10" in reranked[0].text
    assert reranked[0].rank == 1
    assert reranked[0].source == "reranked"


def test_reranker_truncates_to_top_n():
    candidates = [_c(f"passage {i}", i + 1) for i in range(12)]
    assert len(Reranker().rerank("anything", candidates, top_n=8)) == 8


def test_reranking_an_empty_list_is_not_an_error():
    assert Reranker().rerank("anything", []) == []


def test_ranks_are_renumbered_after_reordering():
    candidates = [
        _c("Employees accrue 18 days of annual leave per year.", 1),
        _c("50ml amber glass jars are priced at Rs 22.10 per unit.", 2),
    ]

    reranked = Reranker().rerank("amber glass price", candidates)

    assert [c.rank for c in reranked] == [1, 2]
