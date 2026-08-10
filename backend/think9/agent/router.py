"""Query routing.

A small fast model classifies, because routing every query through the large model is the
most common and most avoidable cost sink. A deterministic classifier backs it up, so the
system still routes correctly with no API key and when the provider is down.
"""

import re

from think9.config import get_settings
from think9.models import Route

ROUTES: tuple[Route, ...] = (
    "factual_lookup",
    "cross_brand_comparison",
    "policy",
    "decision_archaeology",
    "needs_structured_data",
)

_SYSTEM = (
    "Classify the operations question into exactly one of: "
    + ", ".join(ROUTES)
    + ". Reply with the label only, no punctuation or explanation."
)

# Order matters. Archaeology is tested first so "why did we change our standard policy"
# routes on the "why did we", not on "policy". Structured is tested before the rest so an
# aggregation question is not mistaken for a comparison.
_ARCHAEOLOGY = re.compile(
    r"\bwhy (did|do|was|were|are) we\b|\bwhy was\b|\bdiscontinu|\bkill(ed)?\b", re.IGNORECASE
)
_STRUCTURED = re.compile(
    r"\btotal\b|\bsum\b|\bhow much did we spend\b|\bby vendor\b|\blast quarter\b", re.IGNORECASE
)
_CROSS_BRAND = re.compile(
    r"\bwhich brands?\b|\bacross brands?\b|\bcompare\b|\bboth brands\b", re.IGNORECASE
)
_POLICY = re.compile(r"\bpolicy\b|\bstandard\b|\bclause\b|\bwhat is our\b", re.IGNORECASE)


def classify_deterministic(question: str) -> Route:
    if _ARCHAEOLOGY.search(question):
        return "decision_archaeology"
    if _STRUCTURED.search(question):
        return "needs_structured_data"
    if _CROSS_BRAND.search(question):
        return "cross_brand_comparison"
    if _POLICY.search(question):
        return "policy"
    return "factual_lookup"


def classify(question: str, llm=None) -> Route:
    if llm is None:
        return classify_deterministic(question)
    try:
        label = llm.complete(_SYSTEM, question, model=get_settings().router_model).strip().lower()
    except Exception:  # noqa: BLE001 — any provider failure must degrade, never propagate
        # Rate limit, timeout, bad key, provider outage: the deterministic classifier is
        # a complete substitute, so there is no failure here worth surfacing to the user.
        return classify_deterministic(question)
    return label if label in ROUTES else classify_deterministic(question)
