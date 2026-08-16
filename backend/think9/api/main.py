import os
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from think9.agent.graph import ask as graph_ask
from think9.agent.graph import build_graph
from think9.agent.llm import LLM
from think9.config import get_settings
from think9.gates.digest import gap_digest
from think9.retrieval.embed import Embedder
from think9.retrieval.rerank import Reranker
from think9.retrieval.retriever import Retriever
from think9.store.db import make_pool
from think9.store.repository import Repository


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the models and open the pool before serving any request.

    Doing it lazily on first request meant a container that had recycled downloaded the
    reranker weights while the embedder was already resident, and a 512 MB box killed it
    mid-download — which the caller saw as a 502 on a question rather than as a failed
    deploy. Failing here instead makes a broken boot visible as a broken boot.
    """
    brain = get_brain()
    yield
    brain.pool.close()


app = FastAPI(title="Think9 Brain", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    user_groups: list[str] = []
    user_id: str = "anonymous"

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value.strip()


class Brain:
    """Holds the loaded ONNX models, and a pool to borrow a connection from per request.

    The models are expensive to create, cheap to keep and safe to share, which is why this
    runs on a warm container rather than a serverless function. A connection is none of
    those things. Keeping one for the life of the process meant that when the database
    dropped it — Neon suspends its compute after a few idle minutes — every request from
    then on returned 500 until the container happened to restart, while `/health`, which
    touches nothing, went on answering 200. Borrowing per request also stops concurrent
    questions from sharing one connection and therefore one transaction.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.pool = make_pool(settings.database_url)
        self._embedder = Embedder()
        self._reranker = Reranker()
        self._llm = LLM()

    def ask(self, question: str, user_groups: list[str], user_id: str):
        with self.pool.connection() as conn:
            repo = Repository(conn)
            # Compiling the graph costs about 5 ms against a request that takes seconds,
            # which is a price worth paying to keep the connection request-scoped.
            graph = build_graph(Retriever(conn, self._embedder, self._reranker), repo, self._llm)
            answer = graph_ask(graph, question, user_groups, user_id)
            self._log(repo, question, user_id, answer)
        return answer

    def digest(self, limit: int = 20):
        with self.pool.connection() as conn:
            return gap_digest(Repository(conn), limit)

    def ready(self) -> None:
        """Prove the database is actually reachable. Raises if it is not."""
        with self.pool.connection() as conn:
            conn.execute("SELECT 1")

    @staticmethod
    def _log(repo: Repository, question: str, user_id: str, answer) -> None:
        # Every answer must be reconstructable after the fact, and every refusal is a line
        # in the documentation backlog.
        repo.log_query(
            user_id=user_id,
            question=question,
            route=str(answer.trace.get("route", "unknown")),
            coverage_score=float(answer.trace.get("coverage") or 0.0),
            outcome=answer.outcome,
            answer_text=answer.text,
            citations=[
                {
                    "chunk_id": str(c.chunk_id),
                    "document_title": c.document_title,
                    "heading_path": c.heading_path,
                    "deep_link": c.deep_link,
                    "effective_date": c.effective_date.isoformat(),
                }
                for c in answer.citations
            ],
            as_of=answer.as_of,
            trace=answer.trace,
        )


@lru_cache(maxsize=1)
def get_brain() -> Brain:
    return Brain()


@app.get("/health")
def health() -> dict:
    """Liveness only: is this process up. Deliberately touches nothing.

    Something has to stay this cheap for the keep-awake ping, which runs every ten minutes
    and must not hold the database open — see `/ready` for the question this cannot answer.
    """
    return {"status": "ok"}


BrainDep = Annotated[Brain, Depends(get_brain)]


@app.post("/ask")
def ask_endpoint(request: AskRequest, brain: BrainDep) -> dict:
    answer = brain.ask(request.question, request.user_groups, request.user_id)
    return {
        "answer": answer.text,
        "outcome": answer.outcome,
        "as_of": answer.as_of.isoformat() if answer.as_of else None,
        "citations": [
            {
                "chunk_id": str(c.chunk_id),
                "document_title": c.document_title,
                "heading_path": c.heading_path,
                "deep_link": c.deep_link,
                "effective_date": c.effective_date.isoformat(),
            }
            for c in answer.citations
        ],
        "trace": answer.trace,
    }


@app.get("/digest")
def digest_endpoint(brain: BrainDep) -> dict:
    return {"gaps": brain.digest()}


@app.get("/ready")
def ready_endpoint(brain: BrainDep) -> dict:
    """Readiness: can this process actually answer, database included.

    `/health` returning 200 while every question returned 500 is exactly how a dead
    connection stayed invisible for hours. This is the endpoint to check when the app
    claims to be up and is not.
    """
    brain.ready()
    return {"status": "ready"}
