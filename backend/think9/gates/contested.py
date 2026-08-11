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

# What a question has to be asking about for a conflict in that attribute to matter.
_ASKED_ABOUT = {
    "minimum order quantity": re.compile(
        r"\bminimum order\b|\bMOQ\b|\border quantity\b|\bhow many units\b|\bper colour run\b",
        re.IGNORECASE,
    ),
    "unit price": re.compile(
        r"\bprice\b|\bpay\b|\bcost\b|\brate\b|\bper unit\b|\bcharge\b", re.IGNORECASE
    ),
    "lead time": re.compile(r"\blead time\b|\bhow long\b|\bdelivery time\b", re.IGNORECASE),
    "payment terms": re.compile(r"\bpayment term\b|\bnet \d+\b|\bterms\b", re.IGNORECASE),
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


def detect_contested(question: str, chunks: list[RetrievedChunk]) -> ContestedFinding | None:
    """A conflict only matters if the question is asking about the thing in conflict.

    The Korent spec sheet and contract annexe disagree on minimum order quantity, and both
    are retrieved for any Korent question. Without this check, "what neck finish does the
    jar use?" is answered with "two sources disagree on MOQ" — true, and not the question.
    """
    # A demoted source is not a competing claim, it is a former one. Only live documents
    # can contest each other.
    live = [c for c in chunks if not c.demoted]

    asked = question.lower()

    for attribute, pattern in _ATTRIBUTES.items():
        if not _ASKED_ABOUT[attribute].search(question):
            continue
        by_entity: dict[tuple[str, str], dict[str, str]] = {}
        for chunk in live:
            match = pattern.search(chunk.text)
            if not match:
                continue
            by_entity.setdefault(_entity(chunk), {}).setdefault(
                match.group(1), chunk.document.title
            )
        for (supplier, _brand), values in by_entity.items():
            # The conflict must be about the supplier the question names. Asking for
            # Halden Glass's minimum order quantity retrieves the Korent spec sheet and
            # annexe too — they are the corpus's most MOQ-shaped chunks — and reporting
            # Korent's genuine disagreement in answer to a question about Halden is a
            # true statement about the wrong supplier.
            if supplier not in asked:
                continue
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
