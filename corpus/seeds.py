"""The five seeded facts of spec section 2.2.

Document bodies are generated. These facts are hand-placed, because they are what the
evaluation asserts against.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SeededFact:
    key: str
    probe_question: str
    expected_substrings: tuple[str, ...]
    must_not_contain: tuple[str, ...]
    category: str


SEEDED_FACTS: list[SeededFact] = [
    SeededFact(
        key="amber_glass_price",
        probe_question="What do we pay for 50ml amber glass?",
        expected_substrings=("22.10", "Korent"),
        must_not_contain=("18.40",),
        category="temporal",
    ),
    SeededFact(
        key="korent_moq_contested",
        probe_question="What is Korent's minimum order quantity?",
        expected_substrings=("5,000", "8,000"),
        must_not_contain=(),
        category="contested",
    ),
    SeededFact(
        key="korent_cross_brand",
        probe_question="Which brands buy from Korent, and on what terms?",
        expected_substrings=("Nuvia", "Grove"),
        must_not_contain=(),
        category="cross_brand",
    ),
    SeededFact(
        key="unanswerable_gap",
        probe_question="What is our standard freight insurance excess for sea shipments?",
        expected_substrings=(),
        must_not_contain=(),
        category="unanswerable",
    ),
    SeededFact(
        key="mango_variant_archaeology",
        probe_question="Why did we discontinue the mango variant?",
        expected_substrings=("panel", "Grove"),
        must_not_contain=(),
        category="archaeology",
    ),
]


def fact(key: str) -> SeededFact:
    for candidate in SEEDED_FACTS:
        if candidate.key == key:
            return candidate
    raise KeyError(f"no seeded fact named {key!r}")
