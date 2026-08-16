from datetime import date
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient

from think9.api.main import app, get_brain
from think9.models import Answer, Citation

CITATION = Citation(
    chunk_id=uuid4(),
    document_title="korent-quote-2026-01.md",
    heading_path="Pricing",
    deep_link="https://drive/f1",
    effective_date=date(2026, 1, 8),
)


class StubBrain:
    def __init__(self, database_up: bool = True):
        self.asked: list[tuple[str, list[str]]] = []
        self._database_up = database_up

    def ready(self):
        if not self._database_up:
            raise psycopg.OperationalError("the connection is closed")

    def ask(self, question, user_groups, user_id):
        self.asked.append((question, user_groups))
        return Answer(
            text="Rs 22.10 per unit.",
            outcome="answered",
            citations=(CITATION,),
            as_of=date(2026, 1, 8),
            trace={"route": "factual_lookup", "coverage": 0.99},
        )

    def digest(self, limit=20):
        return [
            {
                "question": "freight insurance excess",
                "route": "factual_lookup",
                "coverage": 0.1,
                "asked_at": "2026-08-10T00:00:00Z",
            }
        ]


def client(brain=None) -> TestClient:
    app.dependency_overrides[get_brain] = lambda: brain or StubBrain()
    return TestClient(app)


def test_health_reports_ok():
    assert client().get("/health").json()["status"] == "ok"


def test_ready_reports_ready_when_the_database_answers():
    assert client().get("/ready").json()["status"] == "ready"


def test_ready_fails_when_the_database_is_gone_even_though_health_passes():
    """The distinction that hid a real outage.

    A dropped connection left `/health` green while every question returned 500, so
    nothing that watched the service noticed. Readiness has to fail where liveness cannot.
    """
    unreachable = client(StubBrain(database_up=False))

    assert unreachable.get("/health").json()["status"] == "ok"
    with pytest.raises(psycopg.OperationalError):
        unreachable.get("/ready")


def test_ask_returns_the_answer_with_citations_and_as_of():
    response = client().post(
        "/ask",
        json={"question": "amber glass price", "user_groups": ["procurement"], "user_id": "u1"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["outcome"] == "answered"
    assert body["as_of"] == "2026-01-08"
    assert body["citations"][0]["deep_link"] == "https://drive/f1"
    assert body["trace"]["route"] == "factual_lookup"


def test_ask_rejects_an_empty_question():
    response = client().post("/ask", json={"question": "   ", "user_groups": [], "user_id": "u1"})
    assert response.status_code == 422


def test_ask_passes_the_callers_groups_through_unchanged():
    """The ACL the caller presents is the ACL retrieval enforces."""
    brain = StubBrain()

    client(brain).post("/ask", json={"question": "q", "user_groups": ["legal"], "user_id": "u1"})

    assert brain.asked == [("q", ["legal"])]


def test_digest_lists_the_documentation_backlog():
    body = client().get("/digest").json()
    assert body["gaps"][0]["question"] == "freight insurance excess"
