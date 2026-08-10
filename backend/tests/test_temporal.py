from datetime import date
from uuid import uuid4

from tests.conftest import make_document
from think9.models import RetrievedChunk
from think9.retrieval.temporal import apply_temporal_authority, as_of_date

OLD = make_document(title="Korent Quote 2024", effective_date=date(2024, 3, 12), is_superseded=True)
NEW = make_document(
    title="Korent Quote 2026", effective_date=date(2026, 1, 8), supersedes_id=OLD.id
)


def _chunk(doc, text, score):
    return RetrievedChunk(
        chunk_id=uuid4(), document=doc, heading_path="Pricing", text=text, score=score
    )


def test_a_superseded_document_is_demoted_below_its_successor():
    """The dead price outscores the live one on similarity. It must still lose."""
    chunks = [_chunk(OLD, "Rs 18.40 per unit", 0.95), _chunk(NEW, "Rs 22.10 per unit", 0.60)]

    result = apply_temporal_authority(chunks, route="factual_lookup")

    assert result[0].document.id == NEW.id
    assert result[1].demoted is True
    assert result[1].demoted_by == NEW.id


def test_demotion_is_disabled_for_decision_archaeology():
    """History is what a "why did we" question is asking about."""
    chunks = [_chunk(OLD, "Rs 18.40 per unit", 0.95), _chunk(NEW, "Rs 22.10 per unit", 0.60)]

    result = apply_temporal_authority(chunks, route="decision_archaeology")

    assert result[0].document.id == OLD.id
    assert all(c.demoted is False for c in result)


def test_a_superseded_document_absent_its_successor_is_still_demoted():
    result = apply_temporal_authority([_chunk(OLD, "Rs 18.40", 0.9)], route="factual_lookup")
    assert result[0].demoted is True
    assert result[0].demoted_by is None


def test_unrelated_documents_are_left_alone():
    other = make_document(title="Leave policy", effective_date=date(2025, 6, 1))
    result = apply_temporal_authority([_chunk(other, "18 days", 0.7)], route="policy")
    assert result[0].demoted is False


def test_as_of_is_the_latest_effective_date_among_undemoted_chunks():
    chunks = apply_temporal_authority(
        [_chunk(OLD, "Rs 18.40", 0.95), _chunk(NEW, "Rs 22.10", 0.60)], route="factual_lookup"
    )
    assert as_of_date(chunks) == date(2026, 1, 8)


def test_as_of_is_none_when_every_chunk_is_demoted():
    chunks = apply_temporal_authority([_chunk(OLD, "Rs 18.40", 0.9)], route="factual_lookup")
    assert as_of_date(chunks) is None


def test_demoted_chunks_sort_below_every_live_chunk_regardless_of_score():
    weak_but_live = make_document(title="Weak", effective_date=date(2025, 1, 1))
    result = apply_temporal_authority(
        [_chunk(OLD, "Rs 18.40", 0.99), _chunk(weak_but_live, "something", 0.05)],
        route="factual_lookup",
    )
    assert result[0].document.title == "Weak"
    assert result[1].demoted is True
