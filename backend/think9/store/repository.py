import json
from datetime import date
from uuid import UUID, uuid4

import psycopg
from pgvector import Vector

from think9.models import Document, Owner, ParsedChunk

_DOC_COLUMNS = """id, source_system, source_id, deep_link, title, doc_type, brand_id,
                  function, author, created_at, effective_date, supersedes_id, acl,
                  sensitive, content_hash, is_superseded"""


class Repository:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def upsert_document(self, doc: Document) -> UUID:
        self.conn.execute(
            f"""INSERT INTO documents ({_DOC_COLUMNS})
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (source_system, source_id) DO UPDATE SET
                    deep_link = EXCLUDED.deep_link, title = EXCLUDED.title,
                    doc_type = EXCLUDED.doc_type, brand_id = EXCLUDED.brand_id,
                    function = EXCLUDED.function, author = EXCLUDED.author,
                    effective_date = EXCLUDED.effective_date,
                    supersedes_id = EXCLUDED.supersedes_id, acl = EXCLUDED.acl,
                    sensitive = EXCLUDED.sensitive, content_hash = EXCLUDED.content_hash""",
            (
                doc.id,
                doc.source_system,
                doc.source_id,
                doc.deep_link,
                doc.title,
                doc.doc_type,
                doc.brand_id,
                doc.function,
                doc.author,
                doc.created_at,
                doc.effective_date,
                doc.supersedes_id,
                list(doc.acl),
                doc.sensitive,
                doc.content_hash,
                doc.is_superseded,
            ),
        )
        self.conn.commit()
        return doc.id

    def get_document(self, doc_id: UUID) -> Document | None:
        row = self.conn.execute(
            f"SELECT {_DOC_COLUMNS} FROM documents WHERE id = %s", (doc_id,)
        ).fetchone()
        return _row_to_document(row) if row else None

    def insert_chunks(
        self, document_id: UUID, chunks: list[ParsedChunk], embeddings: list[list[float]]
    ) -> list[UUID]:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        # Replace rather than append: re-ingesting a changed document must not leave its
        # previous chunks retrievable.
        self.conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
        ids: list[UUID] = []
        for chunk, vector in zip(chunks, embeddings, strict=True):
            chunk_id = uuid4()
            self.conn.execute(
                """INSERT INTO chunks (id, document_id, ordinal, heading_path, text, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s)""",
                (
                    chunk_id,
                    document_id,
                    chunk.ordinal,
                    chunk.heading_path,
                    chunk.text,
                    Vector(vector),
                ),
            )
            ids.append(chunk_id)
        self.conn.commit()
        return ids

    def mark_superseded(self) -> int:
        """Record the reverse of every supersedes link.

        Run after ingestion. Without it the temporal layer can only spot a stale document
        when its successor happens to be retrieved alongside it.
        """
        cursor = self.conn.execute(
            """UPDATE documents SET is_superseded = true
               WHERE id IN (SELECT supersedes_id FROM documents WHERE supersedes_id IS NOT NULL)
                 AND is_superseded = false"""
        )
        self.conn.commit()
        return cursor.rowcount

    def upsert_owner(self, owner: Owner) -> None:
        self.conn.execute(
            """INSERT INTO owners (id, brand_id, function, person_name, contact)
               VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT (brand_id, function) DO UPDATE SET
                 person_name = EXCLUDED.person_name, contact = EXCLUDED.contact""",
            (uuid4(), owner.brand_id, owner.function, owner.person_name, owner.contact),
        )
        self.conn.commit()

    def log_query(
        self,
        *,
        user_id: str,
        question: str,
        route: str,
        coverage_score: float,
        outcome: str,
        answer_text: str,
        citations: list[dict],
        as_of: date | None,
        trace: dict,
    ) -> UUID:
        """Every answer must be reconstructable after the fact."""
        query_id = uuid4()
        self.conn.execute(
            """INSERT INTO query_log (id, user_id, question, route, coverage_score, outcome,
                                      answer_text, citations, as_of, trace)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                query_id,
                user_id,
                question,
                route,
                coverage_score,
                outcome,
                answer_text,
                json.dumps(citations, default=str),
                as_of,
                json.dumps(trace, default=str),
            ),
        )
        self.conn.commit()
        return query_id

    def insert_canon(
        self,
        *,
        question: str,
        answer: str,
        author: str,
        source_query_id: UUID | None,
        effective_date: date,
    ) -> UUID:
        """An owner's reply, captured as a first-class memory entry.

        This is the compounding loop: coverage improves as a by-product of people
        answering questions they were going to be asked anyway.
        """
        canon_id = uuid4()
        self.conn.execute(
            """INSERT INTO canon (id, question, answer, author, source_query_id, effective_date)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (canon_id, question, answer, author, source_query_id, effective_date),
        )
        self.conn.commit()
        return canon_id

    def recent_gaps(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            """SELECT question, route, coverage_score, asked_at FROM query_log
               WHERE outcome IN ('refused', 'routed')
               ORDER BY asked_at DESC LIMIT %s""",
            (limit,),
        ).fetchall()
        return [
            {"question": r[0], "route": r[1], "coverage": r[2], "asked_at": r[3].isoformat()}
            for r in rows
        ]

    def find_owner(self, brand_id: str, function: str) -> Owner | None:
        row = self.conn.execute(
            "SELECT brand_id, function, person_name, contact FROM owners "
            "WHERE brand_id = %s AND function = %s",
            (brand_id, function),
        ).fetchone()
        return Owner(*row) if row else None


def _row_to_document(row: tuple) -> Document:
    return Document(
        id=row[0],
        source_system=row[1],
        source_id=row[2],
        deep_link=row[3],
        title=row[4],
        doc_type=row[5],
        brand_id=row[6],
        function=row[7],
        author=row[8],
        created_at=row[9],
        effective_date=row[10],
        supersedes_id=row[11],
        acl=tuple(row[12]),
        sensitive=row[13],
        content_hash=row[14],
        is_superseded=row[15],
    )
