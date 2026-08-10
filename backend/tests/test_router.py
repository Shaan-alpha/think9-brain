import pytest

from think9.agent.router import classify, classify_deterministic


class StubLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, model: str | None = None) -> str:
        self.calls.append((system, user))
        return self.reply


class ExplodingLLM:
    def complete(self, system: str, user: str, model: str | None = None) -> str:
        raise RuntimeError("provider is down")


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What do we pay for 50ml amber glass?", "factual_lookup"),
        ("Which brands buy from Korent and on what terms?", "cross_brand_comparison"),
        ("Why did we discontinue the mango variant?", "decision_archaeology"),
        ("What is our standard exclusivity clause?", "policy"),
        ("Show me total spend by vendor last quarter", "needs_structured_data"),
    ],
)
def test_deterministic_classifier_covers_all_five_routes(question, expected):
    assert classify_deterministic(question) == expected


def test_model_classification_is_used_when_it_returns_a_known_route():
    llm = StubLLM("cross_brand_comparison")
    assert classify("anything at all", llm) == "cross_brand_comparison"
    assert llm.calls


def test_a_firing_deterministic_pattern_overrules_the_model():
    """The patterns are high precision, and the small model gets this exact case wrong.

    llama-3.1-8b-instant classified "show me total spend by vendor last quarter" as a
    factual lookup, which would send an aggregation query into document retrieval rather
    than declining it.
    """
    llm = StubLLM("factual_lookup")

    assert classify("Show me total spend by vendor last quarter", llm) == "needs_structured_data"
    assert llm.calls == []


def test_the_model_is_only_consulted_when_no_pattern_fires():
    llm = StubLLM("policy")

    assert classify("What do we pay for amber glass?", llm) == "policy"
    assert len(llm.calls) == 1


def test_unrecognised_model_output_falls_back_to_the_deterministic_classifier():
    assert classify("Why did we kill the mango variant?", StubLLM("banana")) == (
        "decision_archaeology"
    )


def test_a_failing_provider_falls_back_rather_than_raising():
    assert classify("What do we pay for amber glass?", ExplodingLLM()) == "factual_lookup"


def test_no_llm_means_deterministic_only():
    assert classify("What do we pay for amber glass?", None) == "factual_lookup"


def test_archaeology_wins_over_a_policy_keyword_in_the_same_sentence():
    assert classify_deterministic("Why did we change our standard exclusivity policy?") == (
        "decision_archaeology"
    )
