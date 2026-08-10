import os
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from dotenv import load_dotenv

from think9.config import get_settings
from think9.models import Document
from think9.store.db import apply_schema, connect

# Dev convenience only. Render supplies real environment variables in production, so
# nothing outside the test and local-dev path depends on a .env file existing.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@pytest.fixture(autouse=True, scope="session")
def _placeholder_database_url():
    """Let pure-compute tests run without a database.

    `Settings` requires DATABASE_URL so a misconfigured deploy fails loudly at startup.
    Tests that touch Postgres use the `conn` fixture and TEST_DATABASE_URL; everything
    else only needs the model names on the same Settings object.
    """
    if not os.environ.get("DATABASE_URL"):
        os.environ["DATABASE_URL"] = "postgresql://placeholder/unused-by-this-test"
    get_settings.cache_clear()
    yield


def make_document(**overrides) -> Document:
    """A valid Document with every field populated. Override only what a test is about."""
    base = Document(
        id=uuid4(),
        source_system="google_drive",
        source_id=f"file-{uuid4()}",
        deep_link="https://drive/x",
        title="Korent Quote",
        doc_type="vendor_quote",
        brand_id="nuvia",
        function="procurement",
        author="arun@think9.test",
        created_at=datetime(2026, 1, 5, tzinfo=UTC),
        effective_date=date(2026, 1, 5),
        supersedes_id=None,
        acl=("procurement",),
        sensitive=False,
        content_hash="hash",
    )
    return replace(base, **overrides) if overrides else base


def embedding(seed: float = 0.1) -> list[float]:
    """A deterministic 384-dim vector. The dimension is a global constraint."""
    return [seed] * 384


@pytest.fixture(scope="session")
def _db_connection():
    """One connection and one schema application for the whole session.

    Reconnecting per test costs several seconds against a hosted Postgres, which is enough
    to stop people running the suite.
    """
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    connection = connect(url)
    apply_schema(connection)
    yield connection
    connection.close()


@pytest.fixture
def conn(_db_connection):
    """The session connection, truncated after each test so tests cannot leak into each other.

    TEST_DATABASE_URL must never point at the branch holding the corpus — this truncates.
    """
    yield _db_connection
    _db_connection.execute("TRUNCATE canon, query_log, chunks, documents, owners CASCADE")
    _db_connection.commit()
