"""When two sources conflict and neither supersedes the other, surface both and ask.

Picking one silently is the failure mode that costs the most trust, because the answer
looks exactly as confident as a correct one.
"""

import re
from dataclasses import dataclass

from think9.models import Owner, RetrievedChunk

_ATTRIBUTES = {
    "minimum order quantity": re.compile(
        r"(?:minimum order quantity|MOQ)\D{0,20}([\d,]+)", re.IGNORECASE
    ),
    "unit price": re.compile(r"(?:Rs|₹)\s*([\d,]+\.\d{2})", re.IGNORECASE),
    "lead time": re.compile(r"lead time\D{0,20}([\d,]+)\s*days", re.IGNORECASE),
    "payment terms": re.compile(r"\bNet\s+(\d+)\b", re.IGNORECASE),
}


@dataclass
class ContestedFinding:
    attribute: str
    values: list[tuple[str, str]]
    arbiter: Owner | None = None


def _entity(chunk: RetrievedChunk) -> tuple[str, str]:
    """The (supplier, brand) a chunk is about. Conflicts are only meaningful within one.

    Supplier scoping alone is too loose in both directions. Without it, every spec sheet
    states a minimum order quantity and two different vendors look like they disagree.
    With supplier alone but not brand, Nuvia's Rs 22.10 for a 50ml jar and Grove's Rs 20.75
    for a 180ml vessel — same vendor, different products — read as a contested price when
    both are simply correct.
    """
    return chunk.document.title.split("-")[0].lower(), chunk.document.brand_id


def detect_contested(chunks: list[RetrievedChunk]) -> ContestedFinding | None:
    # A demoted source is not a competing claim, it is a former one. Only live documents
    # can contest each other.
    live = [c for c in chunks if not c.demoted]

    for attribute, pattern in _ATTRIBUTES.items():
        by_entity: dict[tuple[str, str], dict[str, str]] = {}
        for chunk in live:
            match = pattern.search(chunk.text)
            if not match:
                continue
            by_entity.setdefault(_entity(chunk), {}).setdefault(
                match.group(1), chunk.document.title
            )
        for values in by_entity.values():
            if len(values) > 1:
                return ContestedFinding(
                    attribute=attribute,
                    values=[(value, title) for value, title in values.items()],
                )
    return None


def describe(finding: ContestedFinding) -> str:
    sources = "; ".join(f"{value} ({title})" for value, title in finding.values)
    text = (
        f"Two current sources disagree on {finding.attribute}, and neither supersedes the "
        f"other: {sources}."
    )
    if finding.arbiter is not None:
        text += (
            f" {finding.arbiter.person_name} ({finding.arbiter.contact}) owns this and "
            "needs to settle it."
        )
    return text
