from datetime import date
from uuid import uuid4

from tests.conftest import make_document
from think9.agent.verifier import verify
from think9.models import RetrievedChunk

CHUNK_ID = uuid4()
GROVE_CHUNK_ID = uuid4()
DOC = make_document(effective_date=date(2026, 1, 8))
GROVE_DOC = make_document(title="Korent Quote — Grove", brand_id="grove")
CHUNKS = [
    RetrievedChunk(
        chunk_id=CHUNK_ID,
        document=DOC,
        heading_path="Pricing",
        text="50ml amber glass is Rs 22.10 per unit at 5,000 units.",
        score=0.9,
    ),
    RetrievedChunk(
        chunk_id=GROVE_CHUNK_ID,
        document=GROVE_DOC,
        heading_path="Pricing",
        text="180ml amber glass vessel is Rs 20.75 per unit at 10,000 units.",
        score=0.7,
    ),
]


class YesLLM:
    def complete(self, system, user, model=None):
        return "SUPPORTED"


class NoLLM:
    def complete(self, system, user, model=None):
        return "NOT_SUPPORTED"


class ExplodingLLM:
    def complete(self, system, user, model=None):
        raise RuntimeError("provider is down")


def test_a_grounded_claim_survives():
    draft = f"Amber glass costs Rs 22.10 per unit [c:{CHUNK_ID}]."

    result = verify(draft, CHUNKS, YesLLM())

    assert result.refused is False
    assert "22.10" in result.text
    assert result.stripped == []


def test_a_fabricated_number_is_stripped_without_any_model_call():
    """The cheap deterministic check runs first, so a hallucinated price never reaches
    the entailment stage."""
    draft = f"Amber glass costs Rs 31.75 per unit [c:{CHUNK_ID}]."

    result = verify(draft, CHUNKS, llm=None)

    assert "31.75" not in result.text
    assert result.stripped == [draft]
    assert result.refused is True


def test_an_invalid_citation_is_stripped():
    draft = f"Amber glass costs Rs 22.10 per unit [c:{uuid4()}]."

    result = verify(draft, CHUNKS, YesLLM())

    assert result.refused is True
    assert any("citation" in v.reason for v in result.claims)


def test_an_uncited_claim_is_stripped():
    result = verify("Amber glass is expensive this year.", CHUNKS, YesLLM())

    assert result.refused is True
    assert any("no citation" in v.reason for v in result.claims)


def test_a_correctly_sourced_but_wrongly_combined_claim_fails_entailment():
    """Every number is real, the citation resolves, and the claim is still false.

    Rs 20.75 is the Grove 180ml rate; attaching it to the 50ml Nuvia jar recombines two
    genuine facts into one the evidence never asserts. Every deterministic check passes:
    the citation resolves, and 20.75 does appear in the retrieved text. Only entailment
    catches it. This is the gap the Resilience project's error analysis identified and
    could not close.
    """
    draft = f"The 50ml amber glass jar costs Rs 20.75 per unit [c:{CHUNK_ID}]."

    result = verify(draft, CHUNKS, NoLLM())

    assert result.refused is True
    assert any(not v.supported and "entail" in v.reason for v in result.claims)


def test_that_same_claim_passes_every_deterministic_check():
    """Proves the previous test exercises entailment and not a cheaper check."""
    draft = f"The 50ml amber glass jar costs Rs 20.75 per unit [c:{CHUNK_ID}]."

    result = verify(draft, CHUNKS, llm=None)

    assert result.refused is False
    assert "20.75" in result.text


def test_stripping_every_claim_forces_a_refusal():
    draft = f"Rs 99.99 per unit [c:{CHUNK_ID}]. Lead time is 91 days [c:{CHUNK_ID}]."

    result = verify(draft, CHUNKS, YesLLM())

    assert result.refused is True
    assert result.text == ""


def test_a_partially_grounded_draft_keeps_only_the_supported_claims():
    draft = (
        f"Amber glass costs Rs 22.10 per unit [c:{CHUNK_ID}]. Lead time is 91 days [c:{CHUNK_ID}]."
    )

    result = verify(draft, CHUNKS, YesLLM())

    assert result.refused is False
    assert "22.10" in result.text
    assert "91" not in result.text
    assert len(result.stripped) == 1


def test_an_entailment_outage_is_treated_as_unsupported():
    """Fail closed. An unavailable check is not a passed check."""
    draft = f"Amber glass costs Rs 22.10 per unit [c:{CHUNK_ID}]."

    result = verify(draft, CHUNKS, ExplodingLLM())

    assert result.refused is True
    assert any("unavailable" in v.reason for v in result.claims)
