from datetime import date
from uuid import uuid4

from tests.conftest import make_document
from think9.agent.graph import ask, build_graph, infer_scope
from think9.agent.router import classify_deterministic
from think9.models import Owner, RetrievalResult, RetrievedChunk

CHUNK_ID = uuid4()
DOC = make_document(effective_date=date(2026, 1, 8))
GOOD = RetrievalResult(
    chunks=[
        RetrievedChunk(
            chunk_id=CHUNK_ID,
            document=DOC,
            heading_path="Pricing",
            text="50ml amber glass is Rs 22.10 per unit.",
            score=0.82,
        )
    ],
    as_of=date(2026, 1, 8),
    coverage=0.82,
    trace={"dense": [], "sparse": []},
)
EMPTY = RetrievalResult(chunks=[], as_of=None, coverage=0.0, trace={})


class StubRetriever:
    def __init__(self, result):
        self.result = result
        self.calls: list[tuple[str, str]] = []

    def retrieve(self, question, route, user_groups, **kwargs):
        self.calls.append((question, route))
        return self.result


class StubRepo:
    def find_owner(self, brand_id, function):
        return Owner(brand_id, function, "Priya Nair", "priya@think9.test")


class StubLLM:
    """Stands in for a competent model: classifies sensibly, cites, and confirms support."""

    def complete(self, system, user, model=None):
        if "SUPPORTED" in system:
            return "SUPPORTED"
        if "Classify" in system:
            return classify_deterministic(user)
        return f"Amber glass costs Rs 22.10 per unit [c:{CHUNK_ID}]."


def test_a_supported_question_is_answered_with_citations_and_an_as_of_date():
    graph = build_graph(StubRetriever(GOOD), StubRepo(), StubLLM())

    answer = ask(graph, "what do we pay for amber glass", ["procurement"], "u1")

    assert answer.outcome == "answered"
    assert answer.as_of == date(2026, 1, 8)
    assert len(answer.citations) == 1


def test_below_threshold_coverage_routes_to_refusal():
    graph = build_graph(StubRetriever(EMPTY), StubRepo(), StubLLM())

    answer = ask(graph, "what is our freight insurance excess", ["procurement"], "u1")

    assert answer.outcome == "refused"
    assert "Priya Nair" in answer.text


def test_below_threshold_coverage_never_calls_the_synthesiser():
    class CountingLLM(StubLLM):
        def __init__(self):
            self.synthesis_calls = 0

        def complete(self, system, user, model=None):
            if "Cite every factual claim" in system:
                self.synthesis_calls += 1
            return super().complete(system, user, model)

    llm = CountingLLM()
    ask(build_graph(StubRetriever(EMPTY), StubRepo(), llm), "q", ["procurement"], "u1")

    assert llm.synthesis_calls == 0


def test_needs_structured_data_is_declined_gracefully():
    retriever = StubRetriever(GOOD)
    graph = build_graph(retriever, StubRepo(), StubLLM())

    answer = ask(graph, "show me total spend by vendor last quarter", ["procurement"], "u1")

    assert answer.outcome == "refused"
    assert "procurement tables" in answer.text


def test_the_trace_records_the_route_and_every_stage():
    graph = build_graph(StubRetriever(GOOD), StubRepo(), StubLLM())

    answer = ask(graph, "what do we pay for amber glass", ["procurement"], "u1")

    assert answer.trace["route"] == "factual_lookup"
    assert "retrieval" in answer.trace
    assert "verifier" in answer.trace


def test_the_owner_retriever_runs_in_parallel_and_lands_in_the_trace():
    """Both retrievers fan out from the router in one superstep.

    If this ever fails, the conditional edge is suppressing the parallel branch and the
    fan-out needs restructuring around Send.
    """
    graph = build_graph(StubRetriever(GOOD), StubRepo(), StubLLM())

    answer = ask(graph, "what do we pay for amber glass", ["procurement"], "u1")

    assert answer.trace["owner"] == "Priya Nair"


def test_a_draft_that_fails_verification_becomes_a_refusal():
    class UngroundedLLM(StubLLM):
        def complete(self, system, user, model=None):
            if "Cite every factual claim" in system:
                return f"Amber glass costs Rs 99.99 per unit [c:{CHUNK_ID}]."
            return super().complete(system, user, model)

    graph = build_graph(StubRetriever(GOOD), StubRepo(), UngroundedLLM())

    answer = ask(graph, "what do we pay for amber glass", ["procurement"], "u1")

    assert answer.outcome == "refused"
    assert "99.99" not in answer.text


def test_a_contested_fact_diverts_before_synthesis():
    """No model gets the chance to pick a winner between two live, conflicting sources."""
    spec = make_document(title="korent-spec-sheet-2025-11.md")
    annexe = make_document(title="korent-contract-annexe-2025-12.md")
    contested = RetrievalResult(
        chunks=[
            RetrievedChunk(
                uuid4(), spec, "Constraints", "Minimum order quantity: 5,000 units.", 0.9
            ),
            RetrievedChunk(
                uuid4(), annexe, "Clause 4.1", "Minimum order quantity: 8,000 units.", 0.88
            ),
        ],
        as_of=date(2026, 1, 5),
        coverage=0.9,
        trace={},
    )

    class CountingLLM(StubLLM):
        def __init__(self):
            self.synthesis_calls = 0

        def complete(self, system, user, model=None):
            if "Cite every factual claim" in system:
                self.synthesis_calls += 1
            return super().complete(system, user, model)

    llm = CountingLLM()
    answer = ask(
        build_graph(StubRetriever(contested), StubRepo(), llm),
        "What is Korent's minimum order quantity?",
        ["procurement"],
        "u1",
    )

    assert answer.outcome == "contested"
    assert "5,000" in answer.text
    assert "8,000" in answer.text
    assert "Priya Nair" in answer.text
    assert llm.synthesis_calls == 0


def test_scope_inference_reads_the_brand_and_function_from_the_question():
    assert infer_scope("What does Grove pay for glass?") == ("grove", "procurement")
    assert infer_scope("What is Nuvia's creator exclusivity window?") == ("nuvia", "brand_ops")
    assert infer_scope("What do we pay for amber glass?") == ("shared", "procurement")
