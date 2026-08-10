from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(database_url: str) -> psycopg.Connection:
    """Open a connection with the pgvector types registered.

    The extension is created before `register_vector` because registration looks up the
    `vector` type OID, which does not exist until the extension does.
    """
    conn = psycopg.connect(database_url, autocommit=False)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
