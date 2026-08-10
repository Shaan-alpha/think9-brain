"""Dense and sparse retrieval, both filtered by access control in SQL.

Access control is enforced here, at retrieval time, not after generation. The model is
never shown a chunk the requesting user could not open themselves — filtering after
generation leaks, filtering before retrieval cannot.

The two arms exist because a business corpus is entity-heavy. Vendor names, SKU codes,
clause numbers and brand names are exact tokens where embeddings blur and keyword search
excels; everything else is the reverse.
"""

import psycopg
from pgvector import Vector

from think9.models import Candidate

_DENSE_SQL = """
    SELECT c.id, c.document_id, c.text, c.heading_path,
           1 - (c.embedding <=> %(vec)s) AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE d.acl && %(groups)s::text[]
    ORDER BY c.embedding <=> %(vec)s
    LIMIT %(limit)s
"""

_SPARSE_SQL = """
    SELECT c.id, c.document_id, c.text, c.heading_path,
           ts_rank_cd(c.tsv, websearch_to_tsquery('english', %(q)s)) AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE d.acl && %(groups)s::text[]
      AND c.tsv @@ websearch_to_tsquery('english', %(q)s)
    ORDER BY score DESC
    LIMIT %(limit)s
"""


def dense_search(
    conn: psycopg.Connection, query_vector: list[float], user_groups: list[str], limit: int = 30
) -> list[Candidate]:
    rows = conn.execute(
        _DENSE_SQL,
        {"vec": Vector(query_vector), "groups": list(user_groups), "limit": limit},
    ).fetchall()
    return _to_candidates(rows, "dense")


def sparse_search(
    conn: psycopg.Connection, question: str, user_groups: list[str], limit: int = 30
) -> list[Candidate]:
    rows = conn.execute(
        _SPARSE_SQL,
        {"q": question, "groups": list(user_groups), "limit": limit},
    ).fetchall()
    return _to_candidates(rows, "sparse")


def _to_candidates(rows: list[tuple], source: str) -> list[Candidate]:
    return [
        Candidate(
            chunk_id=row[0],
            document_id=row[1],
            text=row[2],
            heading_path=row[3],
            score=float(row[4]),
            rank=index + 1,
            source=source,
        )
        for index, row in enumerate(rows)
    ]
