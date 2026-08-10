from datetime import date

import pytest

from think9.ingest.drive import DriveFile
from think9.ingest.pipeline import (
    MissingProvenance,
    document_id_for,
    ingest,
    parse_front_matter,
    to_document,
)
from think9.models import ParsedChunk
from think9.retrieval.embed import EMBEDDING_DIM
from think9.store.repository import Repository

FILE = DriveFile(
    "f1",
    "korent-quote-2026-01.md",
    "text/markdown",
    "2026-01-08T10:00:00.000Z",
    "https://drive/f1",
)

DOC = """---
brand_id: nuvia
function: procurement
doc_type: vendor_quote
author: arun@think9.test
effective_date: 2026-01-08
supersedes: korent-quote-2024-03.md
acl: [procurement]
sensitive: false
---
# Korent Quote

## Pricing
Rs 22.10 per unit.
"""


def test_front_matter_is_split_from_the_body():
    meta, body = parse_front_matter(DOC)

    assert meta["brand_id"] == "nuvia"
    assert meta["acl"] == ["procurement"]
    assert meta["sensitive"] is False
    assert body.startswith("# Korent Quote")
    assert "brand_id" not in body


def test_to_document_carries_provenance_and_the_deep_link():
    meta, body = parse_front_matter(DOC)

    doc = to_document(FILE, meta, body, supersedes_index={"korent-quote-2024-03.md": "f0"})

    assert doc.deep_link == "https://drive/f1"
    assert doc.effective_date == date(2026, 1, 8)
    assert doc.acl == ("procurement",)
    assert doc.content_hash != ""


@pytest.mark.parametrize("missing", ["brand_id", "function", "doc_type", "effective_date", "acl"])
def test_a_document_missing_required_provenance_fails_loudly(missing):
    meta, body = parse_front_matter(DOC)
    del meta[missing]

    with pytest.raises(MissingProvenance, match=missing):
        to_document(FILE, meta, body, supersedes_index={})


def test_absent_supersedes_is_not_an_error():
    meta, body = parse_front_matter(DOC)
    del meta["supersedes"]

    assert to_document(FILE, meta, body, supersedes_index={}).supersedes_id is None


def test_supersedes_filename_resolves_through_the_name_index():
    """Front matter names its predecessor by filename; ids derive from the source file id.

    Resolving through a name->file-id index built from the folder listing is what keeps
    the two in step. If this breaks, the temporal demo silently stops working while every
    other test still passes.
    """
    meta, body = parse_front_matter(DOC)
    index = {"korent-quote-2024-03.md": "f0"}

    doc = to_document(FILE, meta, body, supersedes_index=index)

    assert doc.supersedes_id == document_id_for("f0")


def test_an_unresolvable_supersedes_target_fails_loudly():
    meta, body = parse_front_matter(DOC)

    with pytest.raises(MissingProvenance, match="korent-quote-2024-03.md"):
        to_document(FILE, meta, body, supersedes_index={"something-else.md": "f9"})


class FakeClient:
    def __init__(self, files: dict[str, str]):
        self.files = files

    def list_folder(self, folder_id):
        return [
            DriveFile(f"id-{name}", name, "text/markdown", "t", f"https://drive/{name}")
            for name in self.files
        ]

    def fetch(self, file):
        return self.files[file.name].encode("utf-8")


class FakeEmbedder:
    def embed_chunks(self, chunks: list[ParsedChunk]) -> list[list[float]]:
        return [[0.1] * EMBEDDING_DIM for _ in chunks]


def _doc(effective: str, supersedes: str = "null") -> str:
    return f"""---
brand_id: nuvia
function: procurement
doc_type: vendor_quote
author: arun@think9.test
effective_date: {effective}
supersedes: {supersedes}
acl: [procurement]
sensitive: false
---
# Korent Quote

## Pricing
Rs 22.10 per unit.
"""


def test_ingest_stores_documents_and_chunks(conn):
    repo = Repository(conn)
    client = FakeClient({"a.md": _doc("2026-01-08")})

    report = ingest(client, repo, FakeEmbedder(), "folder")

    assert report.ingested == 1
    assert report.failures == []
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1


def test_ingest_links_the_supersedes_chain_and_marks_the_predecessor(conn):
    repo = Repository(conn)
    client = FakeClient(
        {
            "korent-quote-2024-03.md": _doc("2024-03-12"),
            "korent-quote-2026-01.md": _doc("2026-01-08", "korent-quote-2024-03.md"),
        }
    )

    ingest(client, repo, FakeEmbedder(), "folder")

    old = repo.get_document(document_id_for("id-korent-quote-2024-03.md"))
    new = repo.get_document(document_id_for("id-korent-quote-2026-01.md"))
    assert new.supersedes_id == old.id
    assert old.is_superseded is True
    assert new.is_superseded is False


def test_reingesting_unchanged_content_is_skipped(conn):
    repo = Repository(conn)
    client = FakeClient({"a.md": _doc("2026-01-08")})

    ingest(client, repo, FakeEmbedder(), "folder")
    second = ingest(client, repo, FakeEmbedder(), "folder")

    assert second.ingested == 0
    assert second.skipped_unchanged == 1


def test_a_document_with_bad_provenance_is_recorded_not_swallowed(conn):
    repo = Repository(conn)
    client = FakeClient({"bad.md": "# No front matter at all\n\n## S\nbody\n"})

    report = ingest(client, repo, FakeEmbedder(), "folder")

    assert report.ingested == 0
    assert len(report.failures) == 1
    assert "bad.md" in report.failures[0][0]


def test_a_crlf_document_ingests(conn):
    """Windows-authored files and Google Docs exports both arrive as CRLF.

    Every regex downstream anchors on \\n, so without normalisation the front matter never
    matches and the document fails provenance — which is exactly what happened on the
    first real corpus run.
    """
    repo = Repository(conn)
    client = FakeClient({"crlf.md": _doc("2026-01-08").replace("\n", "\r\n")})

    report = ingest(client, repo, FakeEmbedder(), "folder")

    assert report.failures == []
    assert report.ingested == 1


def test_one_bad_document_does_not_stop_the_rest(conn):
    repo = Repository(conn)
    client = FakeClient({"bad.md": "# broken\n", "good.md": _doc("2026-01-08")})

    report = ingest(client, repo, FakeEmbedder(), "folder")

    assert report.ingested == 1
    assert len(report.failures) == 1
