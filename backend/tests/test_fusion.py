from uuid import uuid4

from think9.models import Candidate
from think9.retrieval.fusion import reciprocal_rank_fusion

A, B, C = uuid4(), uuid4(), uuid4()


def _c(chunk_id, rank, source):
    return Candidate(
        chunk_id=chunk_id,
        document_id=uuid4(),
        text=str(chunk_id),
        heading_path="h",
        score=1.0 / rank,
        rank=rank,
        source=source,
    )


def test_a_chunk_ranked_by_both_arms_beats_one_ranked_by_only_one():
    dense = [_c(A, 1, "dense"), _c(B, 2, "dense")]
    sparse = [_c(B, 1, "sparse"), _c(C, 2, "sparse")]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert fused[0].chunk_id == B


def test_fusion_scores_follow_the_rrf_formula():
    fused = reciprocal_rank_fusion([[_c(A, 1, "dense")]], k=60)
    assert fused[0].score == 1 / 61


def test_ranks_are_renumbered_from_one_and_marked_fused():
    fused = reciprocal_rank_fusion([[_c(A, 1, "dense"), _c(B, 2, "dense")]])
    assert [c.rank for c in fused] == [1, 2]
    assert all(c.source == "fused" for c in fused)


def test_empty_rankings_produce_no_candidates():
    assert reciprocal_rank_fusion([[], []]) == []


def test_fusion_does_not_need_the_two_arms_scores_to_be_comparable():
    """A cosine and a ts_rank_cd are not on the same scale and never will be.

    RRF uses rank position only, so a sparse arm scoring in the thousandths cannot be
    drowned out by a dense arm scoring near one.
    """
    dense = [_c(A, 1, "dense")]
    sparse = [
        Candidate(
            chunk_id=B,
            document_id=uuid4(),
            text="b",
            heading_path="h",
            score=0.0001,
            rank=1,
            source="sparse",
        )
    ]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert {c.chunk_id for c in fused} == {A, B}
    assert fused[0].score == fused[1].score
