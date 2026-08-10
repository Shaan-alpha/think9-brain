"""Chunk boundaries come from the document's own structure, never a character grid.

Carried from the Resilience project's loaders: overlap exists to repair boundaries you
invented, and here there are none.
"""

import io
import re

from think9.models import ParsedChunk

_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_H2 = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_TURN = re.compile(r"^\[(\d{2}:\d{2})\]\s+([^:]+):\s*(.*)$", re.MULTILINE)


def parse(text: str, doc_type: str) -> list[ParsedChunk]:
    if doc_type == "slack_thread":
        return _parse_slack(text)
    return _parse_sectioned(text)


def _parse_sectioned(text: str) -> list[ParsedChunk]:
    title_match = _H1.search(text)
    title = title_match.group(1).strip() if title_match else "document"

    sections = list(_H2.finditer(text))
    if not sections:
        body = _H1.sub("", text).strip()
        return [ParsedChunk(ordinal=0, heading_path=title, text=body)] if body else []

    chunks: list[ParsedChunk] = []
    for ordinal, match in enumerate(sections):
        start = match.end()
        end = sections[ordinal + 1].start() if ordinal + 1 < len(sections) else len(text)
        body = text[start:end].strip()
        chunks.append(
            ParsedChunk(
                ordinal=ordinal,
                heading_path=f"{title} > {match.group(1).strip()}",
                text=body,
            )
        )
    return chunks


def _parse_slack(text: str) -> list[ParsedChunk]:
    return [
        ParsedChunk(
            ordinal=ordinal,
            heading_path=f"thread > {m.group(2).strip()} {m.group(1)}",
            text=m.group(3).strip(),
        )
        for ordinal, m in enumerate(_TURN.finditer(text))
    ]


def parse_pdf(data: bytes) -> list[ParsedChunk]:
    """A PDF without the heading convention degrades to one chunk per page.

    Coarser than a section, but still a unit a human can go and verify — which beats
    inventing boundaries and citing them confidently.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    extracted = "\n".join(pages)
    if _H2.search(extracted):
        return _parse_sectioned(extracted)
    return [
        ParsedChunk(ordinal=i, heading_path=f"PAGE-{i + 1}", text=page)
        for i, page in enumerate(pages)
        if page
    ]
