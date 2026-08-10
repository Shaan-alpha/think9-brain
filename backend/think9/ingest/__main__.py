"""Run ingestion: `uv run python -m think9.ingest`.

Uses the live Drive connector when GOOGLE_CREDENTIALS_JSON is set, and the local corpus
mirror otherwise. The two clients share an interface, so this is the only line that knows
which is which.
"""

import sys
from pathlib import Path

from think9.config import get_settings
from think9.ingest.drive import DriveClient, LocalFolderClient, build_service
from think9.ingest.pipeline import ingest
from think9.models import Owner
from think9.retrieval.embed import Embedder
from think9.store.db import apply_schema, connect
from think9.store.repository import Repository

CORPUS_MIRROR = Path(__file__).resolve().parents[3] / "corpus" / "out"

# SYNTHETIC. Who a refusal routes to. Without these a refusal is honest but not useful.
OWNERS = (
    Owner("nuvia", "procurement", "Priya Nair", "priya@think9.test"),
    Owner("grove", "procurement", "Arun Menon", "arun@think9.test"),
    Owner("shared", "procurement", "Arun Menon", "arun@think9.test"),
    Owner("nuvia", "brand_ops", "Meera Rao", "meera@think9.test"),
    Owner("grove", "brand_ops", "Meera Rao", "meera@think9.test"),
    Owner("shared", "brand_ops", "Meera Rao", "meera@think9.test"),
)


def main() -> int:
    settings = get_settings()
    if settings.google_credentials_json:
        client = DriveClient(build_service(settings.google_credentials_json))
        source = f"Google Drive folder {settings.drive_folder_id}"
    else:
        client = LocalFolderClient(CORPUS_MIRROR)
        source = f"local mirror {CORPUS_MIRROR} (GOOGLE_CREDENTIALS_JSON unset)"

    print(f"ingesting from {source}")
    conn = connect(settings.database_url)
    apply_schema(conn)
    repo = Repository(conn)
    for owner in OWNERS:
        repo.upsert_owner(owner)

    report = ingest(client, repo, Embedder(), settings.drive_folder_id)

    documents = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    superseded = conn.execute("SELECT count(*) FROM documents WHERE is_superseded").fetchone()[0]
    conn.close()

    print(
        f"ingested={report.ingested} skipped_unchanged={report.skipped_unchanged} "
        f"failures={len(report.failures)}"
    )
    for name, error in report.failures:
        print(f"  FAILED {name}: {error}")
    print(f"documents={documents} chunks={chunks} superseded={superseded}")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
