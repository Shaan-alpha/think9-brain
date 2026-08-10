"""Reciprocal rank fusion.

Fuses ranked lists without needing their scores to be comparable, which matters because a
cosine similarity and a ts_rank_cd are not on the same scale and never will be. RRF reads
rank position only.
"""

from collections import defaultdict
from uuid import UUID

from think9.models import Candidate


def reciprocal_rank_fusion(rankings: list[list[Candidate]], k: int = 60) -> list[Candidate]:
    scores: dict[UUID, float] = defaultdict(float)
    representative: dict[UUID, Candidate] = {}
    for ranking in rankings:
        for candidate in ranking:
            scores[candidate.chunk_id] += 1.0 / (k + candidate.rank)
            representative.setdefault(candidate.chunk_id, candidate)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[Candidate] = []
    for position, (chunk_id, score) in enumerate(ordered, start=1):
        base = representative[chunk_id]
        fused.append(
            Candidate(
                chunk_id=base.chunk_id,
                document_id=base.document_id,
                text=base.text,
                heading_path=base.heading_path,
                score=score,
                rank=position,
                source="fused",
            )
        )
    return fused
