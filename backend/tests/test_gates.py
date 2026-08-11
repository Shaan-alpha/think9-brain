from dataclasses import replace
from datetime import date
from uuid import uuid4

from tests.conftest import make_document
from think9.gates.contested import detect_contested
from think9.gates.digest import gap_digest
from think9.gates.sensitive import frame_sensitive
from think9.models import Answer, RetrievedChunk
from think9.store.repository import Repository

SPEC = make_document(
    title="korent-spec-sheet-2025-11.md", doc_type="spec_sheet", effective_date=date(2025, 11, 2)
)
ANNEXE = make_document(
    title="korent-contract-annexe-2025-12.md",
    doc_type="contract",
    effective_date=date(2025, 12, 9),
    sensitive=True,
)
OTHER_VENDOR = make_document(title="sundara-caps-spec-shared-2025-11-11.md", doc_type="spec_sheet")

MOQ_Q = "What is Korent's minimum order quantity?"
PRICE_Q = "What do we pay for 50ml amber glass?"


def _chunk(doc, text, heading="Order constraints"):
    return RetrievedChunk(
        chunk_id=uuid4(), document=doc, heading_path=heading, text=text, score=0.8
    )


def test_two_live_sources_disagreeing_on_moq_are_detected():
    finding = detect_contested(
        MOQ_Q,
        [
            _chunk(SPEC, "Minimum order quantity: 5,000 units."),
            _chunk(ANNEXE, "Minimum order quantity: 8,000 units."),
        ],
    )

    assert finding is not None
    assert finding.attribute == "minimum order quantity"
    assert {v for v, _ in finding.values} == {"5,000", "8,000"}


def test_agreeing_sources_are_not_contested():
    assert (
        detect_contested(
            MOQ_Q,
            [
                _chunk(SPEC, "Minimum order quantity: 5,000 units."),
                _chunk(ANNEXE, "Minimum order quantity: 5,000 units."),
            ],
        )
        is None
    )


def test_different_vendors_disagreeing_is_not_a_conflict():
    """Every spec sheet in the corpus states an MOQ. Only the same vendor can conflict."""
    assert (
        detect_contested(
            MOQ_Q,
            [
                _chunk(SPEC, "Minimum order quantity: 5,000 units."),
                _chunk(OTHER_VENDOR, "Minimum order quantity: 15,000 units."),
            ],
        )
        is None
    )


def test_the_same_vendor_serving_two_brands_is_not_a_conflict():
    """Nuvia's Rs 22.10 50ml jar and Grove's Rs 20.75 180ml vessel are both correct.

    Same supplier, different brand, different product. Scoping on supplier alone reported
    this as a contested price on the headline demo question.
    """
    nuvia = make_document(title="korent-quote-2026-01.md", brand_id="nuvia")
    grove = make_document(title="korent-quote-grove-2025-09.md", brand_id="grove")

    assert (
        detect_contested(
            PRICE_Q,
            [
                _chunk(nuvia, "50ml amber glass jar: Rs 22.10 per unit.", "Pricing"),
                _chunk(grove, "180ml amber glass vessel: Rs 20.75 per unit.", "Pricing"),
            ],
        )
        is None
    )


def test_a_demoted_source_does_not_create_a_conflict():
    """A superseded document is not a competing claim; it is a former one."""
    demoted = replace(_chunk(ANNEXE, "Minimum order quantity: 8,000 units."), demoted=True)

    assert (
        detect_contested(MOQ_Q, [_chunk(SPEC, "Minimum order quantity: 5,000 units."), demoted])
        is None
    )


def test_a_conflict_the_question_is_not_about_is_ignored():
    """The Korent spec sheet and annexe disagree on MOQ and are retrieved for any Korent
    question. Answering "what neck finish?" with "two sources disagree on MOQ" is true and
    is not the question."""
    conflicting = [
        _chunk(SPEC, "Minimum order quantity: 5,000 units."),
        _chunk(ANNEXE, "Minimum order quantity: 8,000 units."),
    ]

    assert detect_contested("What neck finish does the Korent jar use?", conflicting) is None
    assert detect_contested(MOQ_Q, conflicting) is not None


def test_another_suppliers_conflict_is_not_reported():
    """Found on the held-out set: asking Halden Glass's MOQ returned Korent's conflict.

    The Korent spec sheet and annexe are the corpus's most MOQ-shaped chunks, so they are
    retrieved for any MOQ question. Their disagreement is genuine and is a true statement
    about the wrong supplier.
    """
    halden = make_document(title="halden-glass-spec-shared-2025-11-11.md", brand_id="shared")
    korent_conflict = [
        _chunk(SPEC, "Minimum order quantity: 5,000 units."),
        _chunk(ANNEXE, "Minimum order quantity: 8,000 units."),
        _chunk(halden, "Minimum order quantity: 6,000 units."),
    ]

    assert (
        detect_contested("What is the minimum order quantity for Halden Glass?", korent_conflict)
        is None
    )
    assert detect_contested(MOQ_Q, korent_conflict) is not None


def test_a_single_source_is_not_contested():
    assert detect_contested(MOQ_Q, [_chunk(SPEC, "Minimum order quantity: 5,000 units.")]) is None


def test_a_sensitive_document_forces_evidence_framing():
    answer = Answer(text="The exclusivity term is 12 months.", outcome="answered")

    framed = frame_sensitive(answer, [_chunk(ANNEXE, "Exclusivity: 12 months.")])

    assert framed.text.startswith("Based on the sources below")
    assert framed.outcome == "answered"


def test_a_non_sensitive_answer_is_returned_unchanged():
    answer = Answer(text="Rs 22.10 per unit.", outcome="answered")
    assert frame_sensitive(answer, [_chunk(SPEC, "Rs 22.10")]) is answer


def test_a_demoted_sensitive_document_does_not_trigger_framing():
    answer = Answer(text="x", outcome="answered")
    demoted = replace(_chunk(ANNEXE, "Exclusivity: 12 months."), demoted=True)

    assert frame_sensitive(answer, [demoted]) is answer


# --- query log, canon and the gap digest -------------------------------------


def test_gap_digest_lists_refusals_and_omits_answers(conn):
    repo = Repository(conn)
    repo.log_query(
        user_id="u1",
        question="freight insurance excess",
        route="factual_lookup",
        coverage_score=0.01,
        outcome="refused",
        answer_text="I don't have this.",
        citations=[],
        as_of=None,
        trace={},
    )
    repo.log_query(
        user_id="u1",
        question="amber glass price",
        route="factual_lookup",
        coverage_score=0.99,
        outcome="answered",
        answer_text="Rs 22.10",
        citations=[],
        as_of=date(2026, 1, 8),
        trace={},
    )

    gaps = gap_digest(repo)

    assert [g["question"] for g in gaps] == ["freight insurance excess"]


def test_an_owner_reply_becomes_retrievable_canon(conn):
    """The compounding loop: coverage improves as a by-product of answering questions
    people were going to be asked anyway."""
    repo = Repository(conn)
    query_id = repo.log_query(
        user_id="u1",
        question="What is our freight insurance excess?",
        route="factual_lookup",
        coverage_score=0.01,
        outcome="routed",
        answer_text="",
        citations=[],
        as_of=None,
        trace={},
    )

    canon_id = repo.insert_canon(
        question="What is our freight insurance excess?",
        answer="USD 500 per shipment, per the 2026 marine policy.",
        author="arun@think9.test",
        source_query_id=query_id,
        effective_date=date(2026, 8, 10),
    )

    row = conn.execute(
        "SELECT question, answer, source_query_id FROM canon WHERE id = %s", (canon_id,)
    ).fetchone()
    assert row[1].startswith("USD 500")
    assert row[2] == query_id
