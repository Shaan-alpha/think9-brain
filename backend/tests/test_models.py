import dataclasses
from datetime import UTC, date, datetime
from uuid import uuid4

import pytest

from think9.models import Answer, Candidate, Document, ParsedChunk


def _document(**overrides) -> Document:
    base = {
        "id": uuid4(),
        "source_system": "google_drive",
        "source_id": "file-1",
        "deep_link": "https://drive.google.com/file/d/file-1",
        "title": "Korent Quote 2026",
        "doc_type": "vendor_quote",
        "brand_id": "nuvia",
        "function": "procurement",
        "author": "ops@think9.test",
        "created_at": datetime(2026, 1, 5, tzinfo=UTC),
        "effective_date": date(2026, 1, 5),
        "supersedes_id": None,
        "acl": ("procurement",),
        "sensitive": False,
        "content_hash": "abc123",
    }
    return Document(**{**base, **overrides})


def test_document_is_immutable():
    doc = _document()
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.title = "mutated"


def test_document_defaults_to_not_superseded():
    assert _document().is_superseded is False


def test_parsed_chunk_carries_its_heading_path():
    parsed = ParsedChunk(ordinal=0, heading_path="Pricing > 50ml amber", text="Rs 22.10 per unit")
    assert parsed.heading_path == "Pricing > 50ml amber"


def test_candidate_records_its_rank_and_source():
    candidate = Candidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        text="Rs 22.10 per unit",
        heading_path="Pricing",
        score=0.81,
        rank=1,
        source="dense",
    )
    assert candidate.source == "dense"


def test_answer_defaults_to_no_citations_and_no_as_of():
    answer = Answer(text="I don't have this.", outcome="refused")
    assert answer.citations == ()
    assert answer.as_of is None
    assert answer.trace == {}
