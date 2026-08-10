from dataclasses import replace
from uuid import uuid4

from tests.conftest import make_document
from think9.agent.nodes import synthesise
from think9.models import RetrievedChunk

CHUNK_ID = uuid4()
CHUNKS = [
    RetrievedChunk(
        chunk_id=CHUNK_ID,
        document=make_document(),
        heading_path="Pricing",
        text="50ml amber glass is Rs 22.10 per unit.",
        score=0.9,
    )
]


class EchoLLM:
    def __init__(self):
        self.user_prompt = ""
        self.system_prompt = ""

    def complete(self, system, user, model=None):
        self.system_prompt = system
        self.user_prompt = user
        return f"Amber glass costs Rs 22.10 per unit [c:{CHUNK_ID}]."


def test_synthesis_returns_text_and_resolved_citations():
    text, citations = synthesise(EchoLLM(), "what do we pay for amber glass", CHUNKS)

    assert "22.10" in text
    assert len(citations) == 1
    assert citations[0].chunk_id == CHUNK_ID
    assert citations[0].deep_link == CHUNKS[0].document.deep_link
    assert citations[0].effective_date == CHUNKS[0].document.effective_date


def test_the_prompt_contains_the_chunk_ids_the_model_must_cite():
    llm = EchoLLM()
    synthesise(llm, "q", CHUNKS)
    assert str(CHUNK_ID) in llm.user_prompt


def test_the_context_carries_each_chunks_effective_date():
    llm = EchoLLM()
    synthesise(llm, "q", CHUNKS)
    assert "effective 2026-01-05" in llm.user_prompt


def test_demoted_chunks_are_excluded_from_the_context():
    """A superseded chunk must not reach the model at all, not merely rank lower."""
    llm = EchoLLM()

    synthesise(llm, "q", [replace(CHUNKS[0], demoted=True)])

    assert str(CHUNK_ID) not in llm.user_prompt


def test_only_cited_chunks_become_citations():
    uncited = RetrievedChunk(
        chunk_id=uuid4(),
        document=make_document(title="Unrelated"),
        heading_path="Leave",
        text="18 days annual leave.",
        score=0.4,
    )

    _text, citations = synthesise(EchoLLM(), "q", [*CHUNKS, uncited])

    assert [c.chunk_id for c in citations] == [CHUNK_ID]


def test_the_system_prompt_forbids_ungrounded_numbers():
    llm = EchoLLM()
    synthesise(llm, "q", CHUNKS)
    assert "not appear in the context" in llm.system_prompt
