"""Metrics for the golden-question runs.

A system like this must be judged on numbers rather than demos. Every metric here is
defined over the answer a reader actually receives — the prose together with its
citations — because that is what the claim of groundedness is about.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    """One labelled question."""

    question: str
    category: str
    answerable: bool
    expected_substrings: tuple[str, ...]
    must_not_contain: tuple[str, ...]
    gold_document: str


@dataclass(frozen=True)
class EvalRow:
    """A case, plus what the system did with it."""

    case: EvalCase
    outcome: str
    answer: str
    delivered: str
    gold_retrieved: bool
    claims_supported: int
    claims_total: int


ANSWERING_OUTCOMES = ("answered", "contested")


def _share(numerator: int, denominator: int) -> float:
    # An empty denominator means the metric does not apply to this run, which is not a
    # failure. Reporting 0.0 there would understate a run that simply had no such case.
    return 1.0 if denominator == 0 else numerator / denominator


def is_correct(row: EvalRow) -> bool:
    if not row.case.answerable:
        return row.outcome == "refused"
    if row.outcome not in ANSWERING_OUTCOMES:
        return False
    delivered = row.delivered.lower()
    if any(s.lower() not in delivered for s in row.case.expected_substrings):
        return False
    return all(s.lower() not in delivered for s in row.case.must_not_contain)


def accuracy(rows: list[EvalRow]) -> float:
    return _share(sum(1 for r in rows if is_correct(r)), len(rows))


def groundedness(rows: list[EvalRow]) -> float:
    """Share of claims traceable to a cited span.

    A refusal makes no claims and so cannot be ungrounded; it contributes nothing to
    either side of this ratio.
    """
    return _share(sum(r.claims_supported for r in rows), sum(r.claims_total for r in rows))


def refusal_precision(rows: list[EvalRow]) -> float:
    """When it refused, should it have?"""
    refused = [r for r in rows if r.outcome == "refused"]
    return _share(sum(1 for r in refused if not r.case.answerable), len(refused))


def refusal_recall(rows: list[EvalRow]) -> float:
    """Of the questions it could not support, how many did it decline?"""
    unanswerable = [r for r in rows if not r.case.answerable]
    return _share(sum(1 for r in unanswerable if r.outcome == "refused"), len(unanswerable))


def recall_at_k(rows: list[EvalRow]) -> float:
    scored = [r for r in rows if r.case.answerable and r.case.gold_document]
    return _share(sum(1 for r in scored if r.gold_retrieved), len(scored))


def as_of_correctness(rows: list[EvalRow]) -> float:
    """On temporal questions, did it cite the currently-authoritative fact?

    The failure this measures is the one that destroys trust: a confidently-quoted dead
    price carrying a perfectly valid citation.
    """
    temporal = [r for r in rows if r.case.category == "temporal"]
    return _share(sum(1 for r in temporal if is_correct(r)), len(temporal))


METRICS = {
    "accuracy": accuracy,
    "groundedness": groundedness,
    "refusal_precision": refusal_precision,
    "refusal_recall": refusal_recall,
    "recall_at_k": recall_at_k,
    "as_of_correctness": as_of_correctness,
}


def _summary(rows: list[EvalRow]) -> dict:
    return {"n": len(rows), **{name: round(fn(rows), 4) for name, fn in METRICS.items()}}


def scorecard(rows: list[EvalRow]) -> dict:
    categories = sorted({r.case.category for r in rows})
    return {
        "overall": _summary(rows),
        "by_category": {
            category: _summary([r for r in rows if r.case.category == category])
            for category in categories
        },
    }
