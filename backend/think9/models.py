"""Shared vocabulary. Depends on nothing; imported by everything."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

Route = Literal[
    "factual_lookup",
    "cross_brand_comparison",
    "policy",
    "decision_archaeology",
    "needs_structured_data",
]

Outcome = Literal["answered", "refused", "routed", "contested"]


@dataclass(frozen=True)
class Document:
    id: UUID
    source_system: str
    source_id: str
    deep_link: str
    title: str
    doc_type: str
    brand_id: str
    function: str
    author: str
    created_at: datetime
    effective_date: date
    supersedes_id: UUID | None
    acl: tuple[str, ...]
    sensitive: bool
    content_hash: str
    # Set by the ingest reconciliation pass: true when some other document supersedes this
    # one. Lets the temporal layer identify a stale document without needing its successor
    # to also be in the result set.
    is_superseded: bool = False


@dataclass(frozen=True)
class ParsedChunk:
    """A chunk before it has an identity or an embedding."""

    ordinal: int
    heading_path: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: UUID
    document_id: UUID
    ordinal: int
    heading_path: str
    text: str


@dataclass(frozen=True)
class Candidate:
    """One hit from one retrieval arm, before fusion."""

    chunk_id: UUID
    document_id: UUID
    text: str
    heading_path: str
    score: float
    rank: int
    source: Literal["dense", "sparse", "fused", "reranked"]


@dataclass(frozen=True)
class RetrievedChunk:
    """A candidate joined to its document metadata and judged by the temporal layer."""

    chunk_id: UUID
    document: Document
    heading_path: str
    text: str
    score: float
    demoted: bool = False
    demoted_by: UUID | None = None


@dataclass
class RetrievalResult:
    """Everything the retrieval pipeline learned about one question.

    Lives here rather than in retrieval/ so the agent can depend on the shape without
    depending on the store.
    """

    chunks: list[RetrievedChunk]
    as_of: date | None
    coverage: float
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    chunk_id: UUID
    document_title: str
    heading_path: str
    deep_link: str
    effective_date: date


@dataclass(frozen=True)
class Owner:
    brand_id: str
    function: str
    person_name: str
    contact: str


@dataclass(frozen=True)
class Answer:
    text: str
    outcome: Outcome
    citations: tuple[Citation, ...] = ()
    as_of: date | None = None
    trace: dict[str, Any] = field(default_factory=dict)
