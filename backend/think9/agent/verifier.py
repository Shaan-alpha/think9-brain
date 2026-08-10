"""A separate pass over the draft. Never a clause in the synthesis prompt.

Asking a model to check its own output in the same pass is not a control; it is a
suggestion. The checks run cheapest-first:

1. Citation validity — every [c:<chunk_id>] marker must resolve to a retrieved chunk.
2. Numeric grounding — every digit-string in the claim must appear in the retrieved text.
   Carried from the Resilience project's Gate 3: a price, an MOQ, a lead time and a clause
   number are all digit-strings, so one check covers all four.
3. Entailment — does the evidence actually support the claim as stated? This is what
   catches a claim built from real numbers combined into a relationship the evidence never
   asserts, which the Resilience error analysis identified as the gap deterministic checks
   cannot close.
"""

import re
from dataclasses import dataclass, field

from think9.models import RetrievedChunk

_CITATION = re.compile(r"\[c:([0-9a-fA-F-]{36})\]")
_DIGITS = re.compile(r"\d[\d,.]*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

_ENTAILMENT_SYSTEM = (
    "You check whether a claim is entailed by the evidence. "
    "Reply with exactly SUPPORTED or NOT_SUPPORTED. "
    "A claim that recombines real numbers into a relationship the evidence does not "
    "state is NOT_SUPPORTED."
)


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool
    reason: str


@dataclass
class VerificationResult:
    text: str
    refused: bool
    stripped: list[str] = field(default_factory=list)
    claims: list[ClaimVerdict] = field(default_factory=list)


def verify(draft: str, chunks: list[RetrievedChunk], llm=None) -> VerificationResult:
    valid_ids = {str(c.chunk_id) for c in chunks}
    corpus = " ".join(c.text for c in chunks)

    kept: list[str] = []
    stripped: list[str] = []
    verdicts: list[ClaimVerdict] = []

    for claim in [s.strip() for s in _SENTENCE.split(draft) if s.strip()]:
        verdict = _judge(claim, valid_ids, corpus, llm)
        verdicts.append(verdict)
        (kept if verdict.supported else stripped).append(claim)

    return VerificationResult(
        text=" ".join(kept), refused=not kept, stripped=stripped, claims=verdicts
    )


def _judge(claim: str, valid_ids: set[str], corpus: str, llm) -> ClaimVerdict:
    cited = _CITATION.findall(claim)
    if not cited:
        return ClaimVerdict(claim, False, "no citation")
    if any(cid not in valid_ids for cid in cited):
        return ClaimVerdict(claim, False, "citation does not resolve to a retrieved chunk")

    bare = _CITATION.sub("", claim)
    for number in _DIGITS.findall(bare):
        normalised = number.rstrip(".,")
        if normalised and normalised not in corpus:
            return ClaimVerdict(claim, False, f"ungrounded number {normalised!r}")

    if llm is None:
        return ClaimVerdict(claim, True, "deterministic checks passed; entailment skipped")

    try:
        reply = llm.complete(_ENTAILMENT_SYSTEM, f"EVIDENCE:\n{corpus}\n\nCLAIM:\n{bare}")
    except Exception:  # noqa: BLE001 — fail closed; an unavailable check is not a pass
        return ClaimVerdict(claim, False, "entailment check unavailable")
    if reply.strip().upper().startswith("SUPPORTED"):
        return ClaimVerdict(claim, True, "entailed by evidence")
    return ClaimVerdict(claim, False, "evidence does not entail the claim")
