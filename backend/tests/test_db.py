"""The connection has to survive the database going away underneath it.

Neon suspends its compute after a few minutes without queries, which closes every open
connection. A process that opened one connection at boot and kept the reference served
`psycopg.OperationalError: the connection is closed` on every request afterwards, for as
long as the container happened to live — while `/health`, which touches nothing, kept
answering 200. These tests pin the recovery so that cannot come back.
"""

import os

import pytest
from psycopg_pool import PoolTimeout

from think9.store.db import make_pool


@pytest.fixture
def database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    return url


def test_a_connection_the_server_dropped_does_not_break_the_next_request(database_url):
    """The exact production failure: the connection is closed, and the next caller is fine."""
    with make_pool(database_url, min_size=1, max_size=1) as pool:
        with pool.connection() as first:
            # What Neon does to us after a few idle minutes. Every later use of this same
            # object raises OperationalError("the connection is closed"), which is the
            # error that was reaching users.
            first.close()

        with pool.connection() as second:
            assert second.execute("SELECT 1").fetchone()[0] == 1


def test_the_pool_registers_the_vector_type_on_every_connection(database_url):
    """Registration is per connection, so a replacement connection needs it too.

    Doing it once at startup was survivable only while there was exactly one connection
    that never changed. A pool hands out new ones, and an unregistered `vector` makes
    dense search fail in a way that looks nothing like the cause.
    """
    with make_pool(database_url, min_size=1, max_size=2) as pool:
        with pool.connection() as first:
            first.close()

        with pool.connection() as replacement:
            row = replacement.execute("SELECT '[1,2,3]'::vector").fetchone()
            # A registered vector type comes back as a Vector. Without registration psycopg
            # does not know the OID and hands back the raw text, which has no to_list.
            assert row[0].to_list() == [1.0, 2.0, 3.0]


def test_two_callers_get_two_different_connections(database_url):
    """One shared connection serialised every request and died as a single point of failure."""
    with (
        make_pool(database_url, min_size=2, max_size=2) as pool,
        pool.connection() as first,
        pool.connection() as second,
    ):
        assert first is not second
        assert first.execute("SELECT 1").fetchone()[0] == 1
        assert second.execute("SELECT 1").fetchone()[0] == 1


def test_an_unreachable_database_fails_at_boot_rather_than_on_the_first_question():
    """A broken database must fail the deploy, not the first person who asks something."""
    with pytest.raises(PoolTimeout):
        make_pool(
            "postgresql://nobody@127.0.0.1:1/nothing",
            min_size=1,
            max_size=1,
            wait_timeout=2,
        )
