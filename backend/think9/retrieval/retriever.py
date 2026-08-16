"""The full retrieval pipeline behind one call.

dense + sparse -> reciprocal rank fusion -> cross-encoder rerank -> temporal authority.

The three boolean flags exist so the ablation harness toggles real code paths rather than
reimplementing the pipeline. An ablation that reimplements what it measures proves nothing.
"""

from dataclasses import replace

import psycopg

from think9.models import Candidate, RetrievalResult, RetrievedChunk, Route
from think9.retrieval.embed import Embedder
from think9.retrieval.fusion import reciprocal_rank_fusion
from think9.retrieval.rerank import Reranker
from think9.retrieval.search import dense_search, sparse_search
from think9.retrieval.temporal import apply_temporal_authority, as_of_date
from think9.store.repository import Repository

SHORTLIST = 8


class Retriever:
    def __init__(self, conn: psycopg.Connection, embedder: Embedder, reranker: Reranker) -> None:
        self.conn = conn
        self.embedder = embedder
        self.reranker = reranker
        self.repo = Repository(conn)

    def retrieve(
        self,
        question: str,
        route: Route,
        user_groups: list[str],
        use_hybrid: bool = True,
        use_rerank: bool = True,
        use_temporal: bool = True,
    ) -> RetrievalResult:
        dense = dense_search(self.conn, self.embedder.embed_query(question), user_groups)
        sparse = sparse_search(self.conn, question, user_groups) if use_hybrid else []

        fused = reciprocal_rank_fusion([dense, sparse]) if use_hybrid else _renumber(dense)
        shortlist = (
            self.reranker.rerank(question, fused, top_n=SHORTLIST)
            if use_rerank
            else fused[:SHORTLIST]
        )

        enriched = self._enrich(shortlist)
        judged = apply_temporal_authority(enriched, route) if use_temporal else enriched

        trace = {
            "dense": _summarise(dense),
            "sparse": _summarise(sparse),
            "fused": _summarise(fused),
            "reranked": _summarise(shortlist),
            "demoted": [
                {
                    "title": c.document.title,
                    "effective_date": c.document.effective_date.isoformat(),
                    "demoted_by": str(c.demoted_by) if c.demoted_by else None,
                }
                for c in judged
                if c.demoted
            ],
        }
        return RetrievalResult(
            chunks=judged,
            as_of=as_of_date(judged),
            # Coverage reads the top live chunk. A demoted chunk at the top means the only
            # thing we found is stale, which is not coverage.
            coverage=judged[0].score if judged and not judged[0].demoted else 0.0,
            trace=trace,
        )

    def _enrich(self, candidates: list[Candidate]) -> list[RetrievedChunk]:
        """Attach each candidate's document, in one query for the whole shortlist.

        Taking the shortlist rather than a single candidate is the point: one round trip
        instead of one per chunk. A candidate whose document has gone is dropped, which is
        what the per-chunk version did by returning None.
        """
        documents = self.repo.get_documents([c.document_id for c in candidates])
        return [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                document=documents[c.document_id],
                heading_path=c.heading_path,
                text=c.text,
                score=c.score,
            )
            for c in candidates
            if c.document_id in documents
        ]


def _renumber(candidates: list[Candidate]) -> list[Candidate]:
    return [replace(c, rank=i + 1, source="fused") for i, c in enumerate(candidates)]


def _summarise(candidates: list[Candidate]) -> list[dict]:
    return [
        {
            "chunk_id": str(c.chunk_id),
            "rank": c.rank,
            "score": round(c.score, 4),
            "snippet": c.text[:120],
        }
        for c in candidates[:10]
    ]
