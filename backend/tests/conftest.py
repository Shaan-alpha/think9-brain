import os

import pytest

from think9.config import get_settings


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
