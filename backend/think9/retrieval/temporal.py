"""Business facts decay. A vendor price from 2024 is not the current price.

Without this layer the system confidently quotes a dead price with a perfect citation,
which is exactly the failure that destroys trust. It is the part most retrieval systems
omit and the part that decides whether an operations team still trusts the thing in
week two.
"""

from dataclasses import replace
from datetime import date
from uuid import UUID

from think9.models import RetrievedChunk, Route

DEMOTION_PENALTY = 0.5


def apply_temporal_authority(chunks: list[RetrievedChunk], route: Route) -> list[RetrievedChunk]:
    if route == "decision_archaeology":
        # History is precisely what the question is about. A layer that unconditionally
        # hid superseded documents would break the one query type that needs them.
        return sorted(chunks, key=lambda c: c.score, reverse=True)

    # Map each superseded document to the successor that displaced it, where both happen
    # to be in this result set.
    successor_of: dict[UUID, UUID] = {
        c.document.supersedes_id: c.document.id
        for c in chunks
        if c.document.supersedes_id is not None
    }

    judged: list[RetrievedChunk] = []
    for chunk in chunks:
        successor = successor_of.get(chunk.document.id)
        # `is_superseded` is set at ingest, so a stale document is identifiable even when
        # its successor was not retrieved alongside it.
        if successor is not None or chunk.document.is_superseded:
            judged.append(
                replace(
                    chunk,
                    score=chunk.score * DEMOTION_PENALTY,
                    demoted=True,
                    demoted_by=successor,
                )
            )
        else:
            judged.append(chunk)

    # Live chunks first, then by score. A demoted chunk never outranks a live one, however
    # similar it looked to the embedder.
    return sorted(judged, key=lambda c: (not c.demoted, c.score), reverse=True)


def as_of_date(chunks: list[RetrievedChunk]) -> date | None:
    live = [c.document.effective_date for c in chunks if not c.demoted]
    return max(live) if live else None
