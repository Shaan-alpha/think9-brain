import os
from dataclasses import replace
from datetime import UTC, date, datetime
from uuid import uuid4

import pytest

from think9.config import get_settings
from think9.models import Document


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
