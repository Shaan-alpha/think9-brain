"""Fetch, parse, chunk, embed, store — with provenance validated on the way in.

A document missing required provenance fails loudly rather than entering the index
degraded, because provenance is what makes a citation clickable and an access rule
enforceable. A document that fails is recorded in the report, never silently skipped.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from uuid import NAMESPACE_URL, UUID, uuid5

from think9.ingest.loaders import parse, parse_pdf
from think9.models import Document

REQUIRED = ("brand_id", "function", "doc_type", "effective_date", "acl")
_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


class MissingProvenance(ValueError):
    pass


@dataclass
class IngestReport:
    ingested: int = 0
    skipped_unchanged: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)


def document_id_for(source_id: str) -> UUID:
    """Stable id derived from the source file id, so re-ingesting is idempotent."""
    return uuid5(NAMESPACE_URL, f"google_drive:{source_id}")


def parse_front_matter(text: str) -> tuple[dict, str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    meta: dict = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, raw = line.partition(":")
        meta[key.strip()] = _coerce(raw.strip())
    return meta, text[match.end() :]


def _coerce(raw: str):
    if raw in ("null", ""):
        return None
    if raw in ("true", "false"):
        return raw == "true"
    if raw.startswith("[") and raw.endswith("]"):
        return [item.strip() for item in raw[1:-1].split(",") if item.strip()]
    return raw


def to_document(file, meta: dict, body: str, supersedes_index: dict[str, str]) -> Document:
    """Build a Document, resolving `supersedes` from a filename to a document id.

    Front matter names a predecessor by filename because that is what an author can write.
    Ids derive from the source file id. `supersedes_index` maps one to the other, built
    from the folder listing before any document is converted.
    """
    for key in REQUIRED:
        if meta.get(key) in (None, [], ""):
            raise MissingProvenance(
                f"{file.name}: required provenance field {key!r} is missing. "
                "Provenance is what makes a citation clickable and an ACL enforceable."
            )

    supersedes_name = meta.get("supersedes")
    supersedes_id = None
    if supersedes_name:
        target = supersedes_index.get(supersedes_name)
        if target is None:
            raise MissingProvenance(
                f"{file.name}: supersedes {supersedes_name!r}, which is not in the folder. "
                "An unresolvable lineage would silently disable temporal demotion."
            )
        supersedes_id = document_id_for(target)

    return Document(
        id=document_id_for(file.id),
        source_system="google_drive",
        source_id=file.id,
        deep_link=file.web_view_link,
        title=file.name,
        doc_type=meta["doc_type"],
        brand_id=meta["brand_id"],
        function=meta["function"],
        author=meta.get("author", "unknown"),
        created_at=datetime.now(UTC),
        effective_date=date.fromisoformat(str(meta["effective_date"])),
        supersedes_id=supersedes_id,
        acl=tuple(meta["acl"]),
        sensitive=bool(meta.get("sensitive", False)),
        content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
    )


def ingest(client, repo, embedder, folder_id: str) -> IngestReport:
    report = IngestReport()
    files = client.list_folder(folder_id)
    supersedes_index = {f.name: f.id for f in files}

    for file in files:
        try:
            raw = client.fetch(file)
            if file.mime_type == "application/pdf":
                chunks = parse_pdf(raw)
                meta, body = {}, "\n".join(c.text for c in chunks)
            else:
                # Normalise line endings before anything parses them. Files authored on
                # Windows and Google Docs exports both arrive as CRLF, and every regex
                # downstream — front matter, headings, Slack turn markers — anchors on \n.
                text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
                meta, body = parse_front_matter(text)
                chunks = parse(body, meta.get("doc_type", "transcript"))

            document = to_document(file, meta, body, supersedes_index)
            existing = repo.get_document(document.id)
            if existing and existing.content_hash == document.content_hash:
                report.skipped_unchanged += 1
                continue

            repo.upsert_document(document)
            repo.insert_chunks(document.id, chunks, embedder.embed_chunks(chunks))
            report.ingested += 1
        except Exception as exc:  # noqa: BLE001 — recorded, never silently swallowed
            report.failures.append((file.name, str(exc)))

    # Record the reverse of every lineage link now that all documents exist.
    repo.mark_superseded()
    return report
