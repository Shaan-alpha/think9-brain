import os
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
from think9.store.db import connect
from think9.store.repository import Repository

app = FastAPI(title="Think9 Brain")
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
    """Holds the one long-lived connection and the loaded ONNX models.

    Both are expensive to create and cheap to keep, which is why this runs on a warm
    container rather than a serverless function.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.conn = connect(settings.database_url)
        self.repo = Repository(self.conn)
        self.graph = build_graph(Retriever(self.conn, Embedder(), Reranker()), self.repo, LLM())

    def ask(self, question: str, user_groups: list[str], user_id: str):
        answer = graph_ask(self.graph, question, user_groups, user_id)
        # Every answer must be reconstructable after the fact, and every refusal is a line
        # in the documentation backlog.
        self.repo.log_query(
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
        return answer

    def digest(self, limit: int = 20):
        return gap_digest(self.repo, limit)


@lru_cache(maxsize=1)
def get_brain() -> Brain:
    return Brain()


@app.get("/health")
def health() -> dict:
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
