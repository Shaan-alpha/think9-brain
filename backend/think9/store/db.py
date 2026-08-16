from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(database_url: str) -> psycopg.Connection:
    """Open a connection with the pgvector types registered.

    The extension is created before `register_vector` because registration looks up the
    `vector` type OID, which does not exist until the extension does.

    For one-shot work — ingestion, tests, scripts. A long-lived server wants `make_pool`:
    a connection held across requests is a connection the database will eventually close.
    """
    conn = psycopg.connect(database_url, autocommit=False)
    _prepare(conn)
    return conn


def _prepare(conn: psycopg.Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    # Per connection, not per database: the OID mapping lives on the connection object, so
    # every replacement the pool creates needs it too.
    register_vector(conn)


def make_pool(
    database_url: str,
    min_size: int = 1,
    max_size: int = 4,
    wait_timeout: float = 30.0,
) -> ConnectionPool:
    """A pool that hands out a connection which is known to be working.

    The server closes connections when it feels like it — Neon suspends its compute after
    a few idle minutes and drops everything open. Holding one connection for the life of
    the process meant that the moment the database did this, every subsequent request
    raised `OperationalError: the connection is closed`, permanently, until the container
    happened to restart. `check` costs one round trip per checkout and buys the guarantee
    that this is a transient blip rather than an outage lasting as long as the process.

    `max_size` is small on purpose: the API runs on one 512 MB instance serving a handful
    of concurrent questions, and each idle connection is memory on both ends.
    """
    pool = ConnectionPool(
        database_url,
        min_size=min_size,
        max_size=max_size,
        configure=_prepare,
        check=ConnectionPool.check_connection,
        # Explicit because psycopg_pool plans to flip the default to False.
        open=True,
    )
    try:
        # A database that is unreachable now should fail the deploy, not the first person
        # who asks a question. Without this the pool retries quietly in the background and
        # the failure surfaces as a 500 minutes later.
        pool.wait(timeout=wait_timeout)
    except Exception:
        pool.close()
        raise
    return pool


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
