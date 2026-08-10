from uuid import uuid4

from think9.models import Candidate
from think9.retrieval.rerank import Reranker


def _c(text, rank, heading_path="h"):
    return Candidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        text=text,
        heading_path=heading_path,
        score=0.5,
        rank=rank,
        source="fused",
    )


def test_scores_are_probabilities_not_logits():
    """A raw cross-encoder score is an unbounded logit, so a threshold against it is
    meaningless. Squashing to (0,1) makes tau interpretable: above 0.5 means the reranker
    judges the passage more likely relevant than not."""
    candidates = [
        _c("50ml amber glass jars are priced at Rs 22.10 per unit.", 1),
        _c("The mitochondrion is the powerhouse of the cell.", 2),
    ]

    reranked = Reranker().rerank("what do we pay for amber glass", candidates)

    assert all(0.0 < c.score < 1.0 for c in reranked)
    assert reranked[0].score > 0.5
    assert reranked[-1].score < 0.5


def test_the_heading_path_is_visible_to_the_reranker():
    """The vendor name often lives in the heading, not the chunk body.

    Without this, "What is Korent's MOQ?" reranks a different vendor's spec sheet to the
    top, because every spec sheet's body reads "Minimum order quantity: N units".
    """
    candidates = [
        _c("Minimum order quantity: 15,000 units.", 1, heading_path="Sundara Caps > Constraints"),
        _c(
            "Minimum order quantity: 5,000 units.",
            2,
            heading_path="Korent Glassworks > Constraints",
        ),
    ]

    reranked = Reranker().rerank("What is Korent's minimum order quantity?", candidates)

    assert "5,000" in reranked[0].text


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
