"""Legal, financial and people questions never return a bare answer."""

from dataclasses import replace

from think9.models import Answer, RetrievedChunk

_PREFIX = "Based on the sources below, and framed as evidence rather than a conclusion: "


def frame_sensitive(answer: Answer, chunks: list[RetrievedChunk]) -> Answer:
    if not any(c.document.sensitive for c in chunks if not c.demoted):
        return answer
    return replace(answer, text=_PREFIX + answer.text)
