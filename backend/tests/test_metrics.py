import pytest

from evalkit.metrics import (
    EvalCase,
    EvalRow,
    accuracy,
    as_of_correctness,
    groundedness,
    recall_at_k,
    refusal_precision,
    refusal_recall,
    scorecard,
)


def _case(**overrides) -> EvalCase:
    base = {
        "question": "q",
        "category": "lookup",
        "answerable": True,
        "expected_substrings": (),
        "must_not_contain": (),
        "gold_document": "korent-quote-2026-01.md",
    }
    return EvalCase(**{**base, **overrides})


def _row(case: EvalCase | None = None, **overrides) -> EvalRow:
    base = {
        "case": case or _case(),
        "outcome": "answered",
        "answer": "",
        "delivered": "",
        "gold_retrieved": True,
        "claims_supported": 1,
        "claims_total": 1,
    }
    return EvalRow(**{**base, **overrides})


def test_groundedness_is_the_share_of_claims_with_a_supporting_span():
    rows = [_row(claims_supported=4, claims_total=5), _row(claims_supported=5, claims_total=5)]
    assert groundedness(rows) == 0.9


def test_groundedness_of_a_run_with_no_claims_is_one():
    """A refusal makes no claims, so it cannot be ungrounded."""
    assert groundedness([_row(claims_supported=0, claims_total=0)]) == 1.0


def test_refusal_precision_counts_only_refusals():
    rows = [
        _row(_case(answerable=False), outcome="refused"),
        _row(_case(answerable=True), outcome="refused"),
        _row(_case(answerable=True), outcome="answered"),
    ]
    assert refusal_precision(rows) == 0.5


def test_refusal_recall_counts_only_unanswerable_questions():
    rows = [
        _row(_case(answerable=False), outcome="refused"),
        _row(_case(answerable=False), outcome="answered"),
    ]
    assert refusal_recall(rows) == 0.5


def test_a_run_with_no_refusals_has_perfect_refusal_precision():
    assert refusal_precision([_row(outcome="answered")]) == 1.0


def test_recall_at_k_only_scores_answerable_questions_with_a_gold_document():
    rows = [
        _row(gold_retrieved=True),
        _row(gold_retrieved=False),
        _row(_case(answerable=False, gold_document=""), gold_retrieved=False),
    ]
    assert recall_at_k(rows) == 0.5


def test_as_of_correctness_fails_a_superseded_value():
    temporal = _case(
        category="temporal", expected_substrings=("22.10",), must_not_contain=("18.40",)
    )
    rows = [
        _row(temporal, delivered="Rs 22.10 per unit"),
        _row(temporal, delivered="Rs 18.40 per unit"),
    ]
    assert as_of_correctness(rows) == 0.5


def test_as_of_correctness_ignores_non_temporal_rows():
    assert as_of_correctness([_row()]) == 1.0


def test_accuracy_requires_the_expected_content_and_an_answering_outcome():
    case = _case(expected_substrings=("22.10",))
    assert accuracy([_row(case, outcome="answered", delivered="Rs 22.10")]) == 1.0
    assert accuracy([_row(case, outcome="refused", delivered="Rs 22.10")]) == 0.0


def test_accuracy_treats_a_contested_outcome_as_answering():
    """Surfacing both sides of a genuine conflict is a correct answer, not a failure."""
    case = _case(category="contested", expected_substrings=("5,000", "8,000"))
    row = _row(case, outcome="contested", delivered="5,000 and 8,000 disagree")
    assert accuracy([row]) == 1.0


def test_accuracy_fails_an_answer_containing_a_forbidden_string():
    case = _case(expected_substrings=("22.10",), must_not_contain=("18.40",))
    row = _row(case, outcome="answered", delivered="Rs 22.10, previously Rs 18.40")
    assert accuracy([row]) == 0.0


def test_accuracy_scores_an_unanswerable_question_on_refusing():
    case = _case(answerable=False, gold_document="")
    assert accuracy([_row(case, outcome="refused")]) == 1.0
    assert accuracy([_row(case, outcome="answered")]) == 0.0


def test_scorecard_reports_every_metric_and_a_per_category_breakdown():
    rows = [
        _row(_case(category="lookup", expected_substrings=("a",)), delivered="a"),
        _row(_case(category="unanswerable", answerable=False, gold_document=""), outcome="refused"),
    ]

    card = scorecard(rows)

    assert set(card["overall"]) == {
        "n",
        "accuracy",
        "groundedness",
        "refusal_precision",
        "refusal_recall",
        "recall_at_k",
        "as_of_correctness",
    }
    assert card["overall"]["n"] == 2
    assert card["by_category"]["lookup"]["n"] == 1
    assert card["by_category"]["unanswerable"]["accuracy"] == 1.0


def test_metrics_on_an_empty_run_do_not_divide_by_zero():
    for metric in (
        accuracy,
        groundedness,
        refusal_precision,
        refusal_recall,
        recall_at_k,
        as_of_correctness,
    ):
        assert metric([]) == pytest.approx(1.0)
