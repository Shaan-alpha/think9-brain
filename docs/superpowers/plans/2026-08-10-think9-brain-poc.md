# Think9 Brain POC — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deployed, grounded institutional-memory assistant over a synthetic two-brand corpus that answers with clickable citations and an as-of date, refuses when evidence is insufficient and routes to a named owner, and synthesises across brands — with a held-out scorecard and an ablation table proving the architecture choices.

**Architecture:** A Google Drive connector ingests structure-chunked documents into Neon Postgres with pgvector, carrying full provenance (`effective_date`, `supersedes_id`, `acl`). Retrieval fuses dense (pgvector cosine) and sparse (Postgres full-text) rankings with reciprocal rank fusion, reranks with a local ONNX cross-encoder, then applies a route-aware temporal authority layer. A LangGraph state machine routes the query, retrieves in parallel, synthesises, and passes the draft to a **separate** verifier node that strips unsupported claims or refuses. FastAPI serves it; a Next.js app renders answers plus an inspectable stage-by-stage trace.

**Tech Stack:** Python 3.12 · uv · psycopg 3 + pgvector · fastembed (ONNX, no torch) · LangGraph · FastAPI + uvicorn · pytest + ruff · Next.js 16 + React 19 + Tailwind 4 · Neon Postgres · Render (backend) + Vercel (frontend)

## Global Constraints

Every task's requirements implicitly include this section.

- **Spec:** [`docs/superpowers/specs/2026-08-10-think9-brain-poc-design.md`](../specs/2026-08-10-think9-brain-poc-design.md). Section references below (§2.2, §4.4, …) point there.
- **Python 3.12**, dependencies managed with `uv`. Backend lives in `backend/`.
- **Embedding dimension is 384** (`sentence-transformers/all-MiniLM-L6-v2` via fastembed). Every `vector(384)` column, test fixture and stub must use 384.
- **No torch.** fastembed's ONNX runtime only — this is what keeps the eval reproducible with no API key.
- **TDD, strictly.** Write the failing test, run it and see it fail for the stated reason, write minimal code, run it and see it pass, commit. A step that says "run it and verify it fails" is not ceremonial: if it passes, the test is wrong.
- **Commit messages carry no AI attribution.** No `Co-Authored-By` trailer, no "Generated with" footer. Ever.
- **Lint gate:** `uv run ruff check . && uv run ruff format --check .` passes before every commit.
- **The corpus is synthetic** and must be labelled as such, in bold, in the README and in the web app header. Brands are `nuvia`, `grove`, and `shared`. Functions are `procurement` and `brand_ops`.
- **Secrets** come from environment variables only. Never commit a service-account JSON, a database URL, or an API key. `.gitignore` covers `*.json` credentials, `.env*`.
- **Tests requiring Postgres** read `TEST_DATABASE_URL` and `pytest.skip` when it is unset, so the suite stays runnable on a clean clone.

---

## File Structure

```
backend/
  pyproject.toml               deps, ruff + pytest config
  think9/
    config.py                  env-backed settings
    models.py                  shared dataclasses + Route literal — the vocabulary every other module speaks
    store/
      schema.sql               DDL: documents, chunks, owners, query_log, canon
      db.py                    connection factory, register_vector, migrations runner
      repository.py            typed CRUD; every read takes user_groups and filters ACL in SQL
    ingest/
      loaders.py               bytes + doc_type -> list[ParsedChunk]; boundaries from document structure
      drive.py                 Drive v3 service-account client: list folder, export/download, map metadata
      pipeline.py              fetch -> parse -> chunk -> embed -> upsert, with required-field validation
    retrieval/
      embed.py                 fastembed TextEmbedding wrapper, title-prepend rule
      search.py                dense + sparse SQL, both ACL-filtered
      fusion.py                reciprocal rank fusion (pure)
      rerank.py                fastembed TextCrossEncoder wrapper
      temporal.py              lineage demotion, route-aware (pure)
      retriever.py             composes the five above behind one call
    agent/
      state.py                 BrainState TypedDict + reducers
      llm.py                   OpenAI-compatible client (Groq), one place that talks to a model
      router.py                model classifier + deterministic fallback
      nodes.py                 document/owner retriever nodes, synthesiser, refusal
      verifier.py              deterministic checks then entailment — a separate node, never a prompt clause
      graph.py                 StateGraph wiring
    gates/
      contested.py             conflicting live sources -> surface both, name arbiter
      sensitive.py             sensitive documents -> evidence framing
      digest.py                query_log -> documentation backlog
    api/main.py                FastAPI: /ask (streaming), /trace/{id}, /digest, /health
  tests/                       mirrors think9/ one-to-one
corpus/
  seeds.py                     the five seeded facts of §2.2, as data
  generate.py                  writes corpus/out/ ready for Drive upload
eval/
  questions_dev.csv            ~60, tuning only
  questions_test.csv           ~40, run once
  metrics.py                   groundedness, refusal precision/recall, recall@k, as-of correctness
  run.py                       scorecard
  ablations.py                 the three §7.3 comparisons
web/                           Next.js app; the trace panel is the centrepiece
```

Files that change together live together: retrieval stages sit beside each other because tuning touches all five; agent nodes sit beside the state they mutate. `models.py` is imported everywhere and depends on nothing.

---

## Task 1: Backend scaffold and configuration

**Files:**
- Create: `backend/pyproject.toml`, `backend/think9/__init__.py`, `backend/think9/config.py`, `backend/.env.example`, `.gitignore`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings` dataclass with fields `database_url: str`, `drive_folder_id: str`, `google_credentials_json: str | None`, `llm_api_key: str | None`, `llm_base_url: str`, `llm_model: str`, `router_model: str`, `embedding_model: str`, `reranker_model: str`, `coverage_tau: float`; and `get_settings() -> Settings` (cached).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_config.py
import pytest
from think9.config import Settings, get_settings


def test_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/think9")
    monkeypatch.setenv("DRIVE_FOLDER_ID", "folder-abc")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.database_url == "postgresql://localhost/think9"
    assert settings.drive_folder_id == "folder-abc"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.coverage_tau == 0.5


def test_missing_database_url_fails_loudly(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        get_settings()


def test_coverage_tau_is_overridable(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/think9")
    monkeypatch.setenv("COVERAGE_TAU", "0.62")
    get_settings.cache_clear()

    assert get_settings().coverage_tau == 0.62
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'think9.config'`

- [ ] **Step 3: Write pyproject.toml**

```toml
[project]
name = "think9"
version = "0.1.0"
description = "Think9 Brain — grounded institutional memory over a synthetic multi-brand corpus"
requires-python = ">=3.12"
dependencies = [
    "psycopg[binary]>=3.2",
    "pgvector>=0.3",
    "fastembed>=0.8",
    "numpy>=2.0",
    "langgraph>=1.0",
    "openai>=1.60",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "google-api-python-client>=2.150",
    "google-auth>=2.35",
    "pypdf>=5.0",
    "openpyxl>=3.1",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24", "ruff>=0.8", "httpx>=0.28"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["think9*"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 4: Write config.py**

```python
# backend/think9/config.py
"""Environment-backed settings. One place reads os.environ; everything else takes a Settings."""
import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    database_url: str
    drive_folder_id: str
    google_credentials_json: str | None
    llm_api_key: str | None
    llm_base_url: str
    llm_model: str
    router_model: str
    embedding_model: str
    reranker_model: str
    coverage_tau: float


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example and fill it in.")
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        database_url=_required("DATABASE_URL"),
        drive_folder_id=os.environ.get("DRIVE_FOLDER_ID", ""),
        google_credentials_json=os.environ.get("GOOGLE_CREDENTIALS_JSON"),
        llm_api_key=os.environ.get("LLM_API_KEY"),
        llm_base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        llm_model=os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile"),
        router_model=os.environ.get("ROUTER_MODEL", "llama-3.1-8b-instant"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model="Xenova/ms-marco-MiniLM-L-6-v2",
        coverage_tau=float(os.environ.get("COVERAGE_TAU", "0.5")),
    )
```

Also write `backend/.env.example` listing every variable above with empty values, and a root `.gitignore` containing `.env`, `.env.local`, `*-credentials.json`, `service-account*.json`, `__pycache__/`, `.venv/`, `node_modules/`, `.pytest_cache/`, `corpus/out/`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv sync --extra dev && uv run pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add backend/ .gitignore
git commit -m "Add backend scaffold and environment-backed settings"
```

---

## Task 2: Domain models

**Files:**
- Create: `backend/think9/models.py`
- Test: `backend/tests/test_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the vocabulary every later task uses. `Route` literal; frozen dataclasses `Document`, `Chunk`, `ParsedChunk`, `Candidate`, `RetrievedChunk`, `Citation`, `Owner`, `Answer`. Exact field names below — later tasks depend on them character-for-character.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_models.py
import dataclasses
from datetime import date, datetime, timezone
from uuid import uuid4

import pytest

from think9.models import Answer, Candidate, Chunk, Citation, Document, ParsedChunk


def _document(**overrides) -> Document:
    base = dict(
        id=uuid4(), source_system="google_drive", source_id="file-1",
        deep_link="https://drive.google.com/file/d/file-1", title="Korent Quote 2026",
        doc_type="vendor_quote", brand_id="nuvia", function="procurement",
        author="ops@think9.test", created_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
        effective_date=date(2026, 1, 5), supersedes_id=None,
        acl=("procurement",), sensitive=False, content_hash="abc123",
    )
    return Document(**{**base, **overrides})


def test_document_is_immutable():
    doc = _document()
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.title = "mutated"


def test_parsed_chunk_carries_its_heading_path():
    parsed = ParsedChunk(ordinal=0, heading_path="Pricing > 50ml amber", text="Rs 22.10 per unit")
    assert parsed.heading_path == "Pricing > 50ml amber"


def test_candidate_records_its_rank_and_source():
    candidate = Candidate(
        chunk_id=uuid4(), document_id=uuid4(), text="Rs 22.10 per unit",
        heading_path="Pricing", score=0.81, rank=1, source="dense",
    )
    assert candidate.source == "dense"


def test_answer_defaults_to_no_citations_and_no_as_of():
    answer = Answer(text="I don't have this.", outcome="refused")
    assert answer.citations == ()
    assert answer.as_of is None
    assert answer.trace == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'think9.models'`

- [ ] **Step 3: Write models.py**

```python
# backend/think9/models.py
"""Shared vocabulary. Depends on nothing; imported by everything."""
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

Route = Literal[
    "factual_lookup",
    "cross_brand_comparison",
    "policy",
    "decision_archaeology",
    "needs_structured_data",
]

Outcome = Literal["answered", "refused", "routed", "contested"]


@dataclass(frozen=True)
class Document:
    id: UUID
    source_system: str
    source_id: str
    deep_link: str
    title: str
    doc_type: str
    brand_id: str
    function: str
    author: str
    created_at: datetime
    effective_date: date
    supersedes_id: UUID | None
    acl: tuple[str, ...]
    sensitive: bool
    content_hash: str


@dataclass(frozen=True)
class ParsedChunk:
    """A chunk before it has an identity or an embedding."""
    ordinal: int
    heading_path: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: UUID
    document_id: UUID
    ordinal: int
    heading_path: str
    text: str


@dataclass(frozen=True)
class Candidate:
    """One hit from one retrieval arm, before fusion."""
    chunk_id: UUID
    document_id: UUID
    text: str
    heading_path: str
    score: float
    rank: int
    source: Literal["dense", "sparse", "fused", "reranked"]


@dataclass(frozen=True)
class RetrievedChunk:
    """A candidate joined to its document metadata and judged by the temporal layer."""
    chunk_id: UUID
    document: Document
    heading_path: str
    text: str
    score: float
    demoted: bool = False
    demoted_by: UUID | None = None


@dataclass(frozen=True)
class Citation:
    chunk_id: UUID
    document_title: str
    heading_path: str
    deep_link: str
    effective_date: date


@dataclass(frozen=True)
class Owner:
    brand_id: str
    function: str
    person_name: str
    contact: str


@dataclass(frozen=True)
class Answer:
    text: str
    outcome: Outcome
    citations: tuple[Citation, ...] = ()
    as_of: date | None = None
    trace: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_models.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add backend/think9/models.py backend/tests/test_models.py
git commit -m "Add shared domain models"
```

---

## Task 3: Store — schema, connection, repository

**Files:**
- Create: `backend/think9/store/__init__.py`, `backend/think9/store/schema.sql`, `backend/think9/store/db.py`, `backend/think9/store/repository.py`
- Test: `backend/tests/test_repository.py`, `backend/tests/conftest.py`

**Interfaces:**
- Consumes: `Settings` (Task 1); `Document`, `Chunk`, `Owner` (Task 2).
- Produces: `connect(database_url) -> psycopg.Connection` (vector type registered); `apply_schema(conn)`; `Repository(conn)` with `upsert_document(doc) -> UUID`, `insert_chunks(document_id, chunks: list[ParsedChunk], embeddings: list[list[float]]) -> list[UUID]`, `get_document(doc_id) -> Document | None`, `find_owner(brand_id, function) -> Owner | None`, `log_query(...) -> UUID`, `recent_gaps(limit) -> list[dict]`.

- [ ] **Step 1: Write schema.sql**

```sql
-- backend/think9/store/schema.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id             uuid PRIMARY KEY,
    source_system  text        NOT NULL,
    source_id      text        NOT NULL,
    deep_link      text        NOT NULL,
    title          text        NOT NULL,
    doc_type       text        NOT NULL,
    brand_id       text        NOT NULL,
    function       text        NOT NULL,
    author         text        NOT NULL,
    created_at     timestamptz NOT NULL,
    effective_date date        NOT NULL,
    supersedes_id  uuid        REFERENCES documents(id),
    acl            text[]      NOT NULL,
    sensitive      boolean     NOT NULL DEFAULT false,
    content_hash   text        NOT NULL,
    ingested_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source_system, source_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    id           uuid PRIMARY KEY,
    document_id  uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      int  NOT NULL,
    heading_path text NOT NULL,
    text         text NOT NULL,
    embedding    vector(384) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx       ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS documents_acl_idx    ON documents USING gin (acl);

CREATE TABLE IF NOT EXISTS owners (
    id          uuid PRIMARY KEY,
    brand_id    text NOT NULL,
    function    text NOT NULL,
    person_name text NOT NULL,
    contact     text NOT NULL,
    note        text NOT NULL DEFAULT '',
    UNIQUE (brand_id, function)
);

CREATE TABLE IF NOT EXISTS query_log (
    id             uuid PRIMARY KEY,
    asked_at       timestamptz NOT NULL DEFAULT now(),
    user_id        text        NOT NULL,
    question       text        NOT NULL,
    route          text        NOT NULL,
    coverage_score double precision NOT NULL,
    outcome        text        NOT NULL,
    answer_text    text        NOT NULL DEFAULT '',
    citations      jsonb       NOT NULL DEFAULT '[]'::jsonb,
    as_of          date,
    trace          jsonb       NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS canon (
    id              uuid PRIMARY KEY,
    question        text NOT NULL,
    answer          text NOT NULL,
    author          text NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    source_query_id uuid REFERENCES query_log(id),
    effective_date  date NOT NULL
);
```

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/conftest.py
import os
import uuid

import pytest

from think9.store.db import apply_schema, connect


@pytest.fixture
def conn():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    connection = connect(url)
    apply_schema(connection)
    yield connection
    connection.execute("TRUNCATE canon, query_log, chunks, documents, owners CASCADE")
    connection.commit()
    connection.close()


def embedding(seed: float = 0.1) -> list[float]:
    """A deterministic 384-dim vector. The dimension is a global constraint."""
    return [seed] * 384
```

```python
# backend/tests/test_repository.py
from datetime import date, datetime, timezone
from uuid import uuid4

from tests.conftest import embedding
from think9.models import Document, Owner, ParsedChunk
from think9.store.repository import Repository


def make_document(**overrides) -> Document:
    base = dict(
        id=uuid4(), source_system="google_drive", source_id=f"file-{uuid4()}",
        deep_link="https://drive.google.com/x", title="Korent Quote 2026",
        doc_type="vendor_quote", brand_id="nuvia", function="procurement",
        author="ops@think9.test", created_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
        effective_date=date(2026, 1, 5), supersedes_id=None,
        acl=("procurement",), sensitive=False, content_hash="hash",
    )
    return Document(**{**base, **overrides})


def test_upsert_document_then_read_it_back(conn):
    repo = Repository(conn)
    doc = make_document()

    repo.upsert_document(doc)

    stored = repo.get_document(doc.id)
    assert stored is not None
    assert stored.title == "Korent Quote 2026"
    assert stored.acl == ("procurement",)
    assert stored.effective_date == date(2026, 1, 5)


def test_upsert_is_idempotent_on_source_id(conn):
    repo = Repository(conn)
    doc = make_document(title="First")
    repo.upsert_document(doc)
    repo.upsert_document(make_document(id=doc.id, source_id=doc.source_id, title="Second"))

    assert repo.get_document(doc.id).title == "Second"
    assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 1


def test_insert_chunks_stores_embeddings_and_generates_tsv(conn):
    repo = Repository(conn)
    doc = make_document()
    repo.upsert_document(doc)

    ids = repo.insert_chunks(
        doc.id,
        [ParsedChunk(ordinal=0, heading_path="Pricing", text="Rs 22.10 per unit")],
        [embedding(0.2)],
    )

    assert len(ids) == 1
    row = conn.execute("SELECT tsv IS NOT NULL FROM chunks WHERE id = %s", (ids[0],)).fetchone()
    assert row[0] is True


def test_find_owner_resolves_brand_and_function(conn):
    repo = Repository(conn)
    repo.upsert_owner(Owner("nuvia", "procurement", "Priya Nair", "priya@think9.test"))

    owner = repo.find_owner("nuvia", "procurement")

    assert owner is not None and owner.person_name == "Priya Nair"
    assert repo.find_owner("grove", "procurement") is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && TEST_DATABASE_URL=$TEST_DATABASE_URL uv run pytest tests/test_repository.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'think9.store.db'`. If `TEST_DATABASE_URL` is unset the tests skip; set it to a Neon branch URL before continuing, because skipped tests prove nothing.

- [ ] **Step 4: Write db.py**

```python
# backend/think9/store/db.py
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(database_url: str) -> psycopg.Connection:
    conn = psycopg.connect(database_url, autocommit=False)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
```

- [ ] **Step 5: Write repository.py**

```python
# backend/think9/store/repository.py
from uuid import UUID, uuid4

import psycopg
from pgvector import Vector

from think9.models import Document, Owner, ParsedChunk

_DOC_COLUMNS = """id, source_system, source_id, deep_link, title, doc_type, brand_id,
                  function, author, created_at, effective_date, supersedes_id, acl,
                  sensitive, content_hash"""


class Repository:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def upsert_document(self, doc: Document) -> UUID:
        self.conn.execute(
            f"""INSERT INTO documents ({_DOC_COLUMNS})
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (source_system, source_id) DO UPDATE SET
                    deep_link = EXCLUDED.deep_link, title = EXCLUDED.title,
                    doc_type = EXCLUDED.doc_type, brand_id = EXCLUDED.brand_id,
                    function = EXCLUDED.function, author = EXCLUDED.author,
                    effective_date = EXCLUDED.effective_date,
                    supersedes_id = EXCLUDED.supersedes_id, acl = EXCLUDED.acl,
                    sensitive = EXCLUDED.sensitive, content_hash = EXCLUDED.content_hash""",
            (doc.id, doc.source_system, doc.source_id, doc.deep_link, doc.title, doc.doc_type,
             doc.brand_id, doc.function, doc.author, doc.created_at, doc.effective_date,
             doc.supersedes_id, list(doc.acl), doc.sensitive, doc.content_hash),
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
        self.conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
        ids: list[UUID] = []
        for chunk, vector in zip(chunks, embeddings, strict=True):
            chunk_id = uuid4()
            self.conn.execute(
                """INSERT INTO chunks (id, document_id, ordinal, heading_path, text, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s)""",
                (chunk_id, document_id, chunk.ordinal, chunk.heading_path,
                 chunk.text, Vector(vector)),
            )
            ids.append(chunk_id)
        self.conn.commit()
        return ids

    def upsert_owner(self, owner: Owner) -> None:
        self.conn.execute(
            """INSERT INTO owners (id, brand_id, function, person_name, contact)
               VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT (brand_id, function) DO UPDATE SET
                 person_name = EXCLUDED.person_name, contact = EXCLUDED.contact""",
            (uuid4(), owner.brand_id, owner.function, owner.person_name, owner.contact),
        )
        self.conn.commit()

    def find_owner(self, brand_id: str, function: str) -> Owner | None:
        row = self.conn.execute(
            "SELECT brand_id, function, person_name, contact FROM owners "
            "WHERE brand_id = %s AND function = %s",
            (brand_id, function),
        ).fetchone()
        return Owner(*row) if row else None


def _row_to_document(row: tuple) -> Document:
    return Document(
        id=row[0], source_system=row[1], source_id=row[2], deep_link=row[3], title=row[4],
        doc_type=row[5], brand_id=row[6], function=row[7], author=row[8], created_at=row[9],
        effective_date=row[10], supersedes_id=row[11], acl=tuple(row[12]),
        sensitive=row[13], content_hash=row[14],
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_repository.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add backend/think9/store/ backend/tests/test_repository.py backend/tests/conftest.py
git commit -m "Add Postgres schema, connection factory and typed repository"
```

---

## Task 4: Loaders — structure-derived chunking

**Files:**
- Create: `backend/think9/ingest/__init__.py`, `backend/think9/ingest/loaders.py`
- Test: `backend/tests/test_loaders.py`

**Interfaces:**
- Consumes: `ParsedChunk` (Task 2).
- Produces: `parse(text: str, doc_type: str) -> list[ParsedChunk]`, and `parse_pdf(data: bytes) -> list[ParsedChunk]`.

Boundaries come from the document's own structure — never a character grid. Markdown/Docs split on `## ` headings; Slack exports split on a `[HH:MM] name:` turn marker; sheets split one chunk per row; a PDF without headings degrades to one chunk per page, labelled `PAGE-n`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_loaders.py
from think9.ingest.loaders import parse

DOC = """# Korent Glassworks Quote

## Pricing
50ml amber glass: Rs 22.10 per unit.

## Terms
Net 45 from invoice date.
"""

SLACK = """[09:12] priya: Did we ever confirm Korent's MOQ?
[09:14] arun: The 2025 sheet says 5,000.
[09:15] priya: The contract annexe says 8,000. Someone needs to settle this.
"""


def test_markdown_splits_on_h2_headings():
    chunks = parse(DOC, "vendor_quote")

    assert [c.heading_path for c in chunks] == [
        "Korent Glassworks Quote > Pricing",
        "Korent Glassworks Quote > Terms",
    ]
    assert "Rs 22.10" in chunks[0].text
    assert chunks[0].ordinal == 0 and chunks[1].ordinal == 1


def test_heading_text_is_not_duplicated_into_body():
    chunks = parse(DOC, "vendor_quote")
    assert not chunks[0].text.startswith("## Pricing")


def test_prose_mentioning_a_heading_pattern_does_not_create_a_section():
    text = "# Memo\n\n## Context\nThe row labelled ## Pricing was wrong.\n"
    assert len(parse(text, "decision_memo")) == 1


def test_slack_export_splits_on_turn_markers():
    chunks = parse(SLACK, "slack_thread")

    assert len(chunks) == 3
    assert chunks[1].heading_path == "thread > arun 09:14"
    assert "5,000" in chunks[1].text


def test_document_with_no_structure_returns_one_chunk():
    chunks = parse("A flat paragraph with no headings at all.", "transcript")
    assert len(chunks) == 1
    assert chunks[0].heading_path == "document"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_loaders.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'think9.ingest.loaders'`

- [ ] **Step 3: Write loaders.py**

```python
# backend/think9/ingest/loaders.py
"""Chunk boundaries come from the document's own structure, never a character grid.

Carried from the Resilience project's loaders: overlap exists to repair boundaries you
invented, and here there are none.
"""
import io
import re

from think9.models import ParsedChunk

_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_H2 = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_TURN = re.compile(r"^\[(\d{2}:\d{2})\]\s+([^:]+):\s*(.*)$", re.MULTILINE)


def parse(text: str, doc_type: str) -> list[ParsedChunk]:
    if doc_type == "slack_thread":
        return _parse_slack(text)
    return _parse_sectioned(text)


def _parse_sectioned(text: str) -> list[ParsedChunk]:
    title_match = _H1.search(text)
    title = title_match.group(1).strip() if title_match else "document"

    sections = list(_H2.finditer(text))
    if not sections:
        body = _H1.sub("", text).strip()
        return [ParsedChunk(ordinal=0, heading_path=title, text=body)] if body else []

    chunks: list[ParsedChunk] = []
    for ordinal, match in enumerate(sections):
        start = match.end()
        end = sections[ordinal + 1].start() if ordinal + 1 < len(sections) else len(text)
        body = text[start:end].strip()
        chunks.append(
            ParsedChunk(
                ordinal=ordinal,
                heading_path=f"{title} > {match.group(1).strip()}",
                text=body,
            )
        )
    return chunks


def _parse_slack(text: str) -> list[ParsedChunk]:
    return [
        ParsedChunk(
            ordinal=ordinal,
            heading_path=f"thread > {m.group(2).strip()} {m.group(1)}",
            text=m.group(3).strip(),
        )
        for ordinal, m in enumerate(_TURN.finditer(text))
    ]


def parse_pdf(data: bytes) -> list[ParsedChunk]:
    """A PDF without the heading convention degrades to one chunk per page.

    Coarser than a section, but still a unit a human can go and verify — which beats
    inventing boundaries and citing them confidently.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    if _H2.search(extracted):
        return _parse_sectioned(extracted)
    return [
        ParsedChunk(ordinal=i, heading_path=f"PAGE-{i + 1}", text=(page.extract_text() or "").strip())
        for i, page in enumerate(reader.pages)
        if (page.extract_text() or "").strip()
    ]
```

Note on `test_prose_mentioning_a_heading_pattern_does_not_create_a_section`: `_H2` is anchored with `re.MULTILINE` to line start, so `The row labelled ## Pricing` mid-line cannot match. The document has one real `## Context` section, hence one chunk.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_loaders.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add backend/think9/ingest/ backend/tests/test_loaders.py
git commit -m "Add structure-derived document loaders"
```

---

## Task 5: Embeddings

**Files:**
- Create: `backend/think9/retrieval/__init__.py`, `backend/think9/retrieval/embed.py`
- Test: `backend/tests/test_embed.py`

**Interfaces:**
- Consumes: `Settings` (Task 1), `ParsedChunk` (Task 2).
- Produces: `Embedder` with `embed_chunks(chunks: list[ParsedChunk]) -> list[list[float]]` and `embed_query(question: str) -> list[float]`; module constant `EMBEDDING_DIM = 384`.

The heading path is prepended to the embedded text, because several questions nearly restate their section heading. The raw body alone is what gets quoted and grounded against, so heading text never leaks into an answer.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_embed.py
from think9.models import ParsedChunk
from think9.retrieval.embed import EMBEDDING_DIM, Embedder, embed_input


def test_embed_input_prepends_the_heading_path():
    chunk = ParsedChunk(ordinal=0, heading_path="Korent Quote > Pricing", text="Rs 22.10 per unit")
    assert embed_input(chunk) == "Korent Quote > Pricing\n\nRs 22.10 per unit"


def test_embedding_dimension_matches_the_schema():
    assert EMBEDDING_DIM == 384


def test_embedder_returns_one_vector_per_chunk_of_the_right_width():
    embedder = Embedder()
    chunks = [
        ParsedChunk(ordinal=0, heading_path="A", text="amber glass pricing"),
        ParsedChunk(ordinal=1, heading_path="B", text="payment terms net 45"),
    ]

    vectors = embedder.embed_chunks(chunks)

    assert len(vectors) == 2
    assert all(len(v) == EMBEDDING_DIM for v in vectors)


def test_query_embedding_is_closer_to_the_relevant_chunk():
    embedder = Embedder()
    query = embedder.embed_query("what do we pay for amber glass")
    relevant, irrelevant = embedder.embed_chunks([
        ParsedChunk(0, "Pricing", "50ml amber glass costs Rs 22.10 per unit"),
        ParsedChunk(1, "Leave", "Employees accrue 18 days of annual leave"),
    ])

    assert _cosine(query, relevant) > _cosine(query, irrelevant)


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np
    va, vb = np.array(a), np.array(b)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_embed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'think9.retrieval.embed'`

- [ ] **Step 3: Write embed.py**

```python
# backend/think9/retrieval/embed.py
from functools import lru_cache

from fastembed import TextEmbedding

from think9.config import get_settings
from think9.models import ParsedChunk

EMBEDDING_DIM = 384


def embed_input(chunk: ParsedChunk) -> str:
    """What gets embedded. Not what gets quoted."""
    return f"{chunk.heading_path}\n\n{chunk.text}"


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=get_settings().embedding_model)


class Embedder:
    def embed_chunks(self, chunks: list[ParsedChunk]) -> list[list[float]]:
        texts = [embed_input(c) for c in chunks]
        return [vector.tolist() for vector in _model().embed(texts)]

    def embed_query(self, question: str) -> list[float]:
        return next(iter(_model().embed([question]))).tolist()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_embed.py -v`
Expected: 4 passed. The first run downloads ~90 MB of ONNX weights.

- [ ] **Step 5: Commit**

```bash
git add backend/think9/retrieval/ backend/tests/test_embed.py
git commit -m "Add fastembed embedder with heading-prepended inputs"
```

---

## Task 6: Corpus — seeded facts and generator

**Files:**
- Create: `corpus/seeds.py`, `corpus/generate.py`, `corpus/README.md`
- Test: `backend/tests/test_corpus_seeds.py`

**Interfaces:**
- Consumes: nothing (standalone; the backend imports `corpus.seeds` only in eval).
- Produces: `SEEDED_FACTS: list[SeededFact]` and `generate(out_dir: Path) -> list[Path]`. `SeededFact` fields: `key`, `probe_question`, `expected_substrings: tuple[str, ...]`, `must_not_contain: tuple[str, ...]`, `category`.

These five facts are the contract between the corpus and the eval. Task 19 asserts against them by `key`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_corpus_seeds.py
from corpus.seeds import SEEDED_FACTS, fact


def test_all_five_seeded_facts_are_present():
    assert {f.key for f in SEEDED_FACTS} == {
        "amber_glass_price", "korent_moq_contested", "korent_cross_brand",
        "unanswerable_gap", "mango_variant_archaeology",
    }


def test_the_temporal_fact_forbids_the_superseded_price():
    temporal = fact("amber_glass_price")
    assert "22.10" in temporal.expected_substrings
    assert "18.40" in temporal.must_not_contain
    assert temporal.category == "temporal"


def test_the_gap_fact_expects_a_refusal():
    assert fact("unanswerable_gap").category == "unanswerable"
    assert fact("unanswerable_gap").expected_substrings == ()


def test_archaeology_fact_permits_superseded_evidence():
    archaeology = fact("mango_variant_archaeology")
    assert archaeology.category == "archaeology"
    assert archaeology.must_not_contain == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_corpus_seeds.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'corpus.seeds'`. Add `pythonpath = [".", ".."]` under `[tool.pytest.ini_options]` in `backend/pyproject.toml` so `corpus` resolves from the repo root.

- [ ] **Step 3: Write seeds.py**

```python
# corpus/seeds.py
"""The five seeded facts of spec section 2.2.

Document bodies are generated. These facts are hand-placed, because they are what the
evaluation asserts against.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SeededFact:
    key: str
    probe_question: str
    expected_substrings: tuple[str, ...]
    must_not_contain: tuple[str, ...]
    category: str


SEEDED_FACTS: list[SeededFact] = [
    SeededFact(
        key="amber_glass_price",
        probe_question="What do we pay for 50ml amber glass?",
        expected_substrings=("22.10", "Korent"),
        must_not_contain=("18.40",),
        category="temporal",
    ),
    SeededFact(
        key="korent_moq_contested",
        probe_question="What is Korent's minimum order quantity?",
        expected_substrings=("5,000", "8,000"),
        must_not_contain=(),
        category="contested",
    ),
    SeededFact(
        key="korent_cross_brand",
        probe_question="Which brands buy from Korent, and on what terms?",
        expected_substrings=("Nuvia", "Grove"),
        must_not_contain=(),
        category="cross_brand",
    ),
    SeededFact(
        key="unanswerable_gap",
        probe_question="What is our standard freight insurance excess for sea shipments?",
        expected_substrings=(),
        must_not_contain=(),
        category="unanswerable",
    ),
    SeededFact(
        key="mango_variant_archaeology",
        probe_question="Why did we discontinue the mango variant?",
        expected_substrings=("panel", "Grove"),
        must_not_contain=(),
        category="archaeology",
    ),
]


def fact(key: str) -> SeededFact:
    for candidate in SEEDED_FACTS:
        if candidate.key == key:
            return candidate
    raise KeyError(f"no seeded fact named {key!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_corpus_seeds.py -v`
Expected: 4 passed

- [ ] **Step 5: Write generate.py and author the corpus**

`generate.py` writes markdown files to `corpus/out/`, one per document, with a YAML front-matter block carrying the metadata the connector will need if Drive metadata is unavailable:

```python
# corpus/generate.py
"""Writes the synthetic corpus to corpus/out/ for upload to Drive.

SYNTHETIC DATA. No real Think9 information appears anywhere in this corpus.
"""
from datetime import date
from pathlib import Path
from textwrap import dedent

FRONT_MATTER = """---
brand_id: {brand_id}
function: {function}
doc_type: {doc_type}
author: {author}
effective_date: {effective_date}
supersedes: {supersedes}
acl: {acl}
sensitive: {sensitive}
---
"""


def _write(out_dir: Path, name: str, meta: dict, body: str) -> Path:
    path = out_dir / name
    path.write_text(FRONT_MATTER.format(**meta) + dedent(body).strip() + "\n", encoding="utf-8")
    return path


def generate(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        _write(out_dir, "korent-quote-2024-03.md", dict(
            brand_id="nuvia", function="procurement", doc_type="vendor_quote",
            author="arun@think9.test", effective_date=date(2024, 3, 12),
            supersedes="null", acl="[procurement]", sensitive="false",
        ), """
            # Korent Glassworks Quote — March 2024

            ## Pricing
            50ml amber glass jar, flint-free: Rs 18.40 per unit at 5,000 units.

            ## Terms
            Net 30 from invoice date. Lead time 21 days ex-Kosamba.
        """),
        _write(out_dir, "korent-quote-2026-01.md", dict(
            brand_id="nuvia", function="procurement", doc_type="vendor_quote",
            author="arun@think9.test", effective_date=date(2026, 1, 8),
            supersedes="korent-quote-2024-03.md", acl="[procurement]", sensitive="false",
        ), """
            # Korent Glassworks Quote — January 2026

            ## Pricing
            50ml amber glass jar, flint-free: Rs 22.10 per unit at 5,000 units.

            ## Terms
            Net 45 from invoice date. Lead time 28 days ex-Kosamba.
        """),
    ]
    return written


if __name__ == "__main__":
    paths = generate(Path(__file__).parent / "out")
    print(f"wrote {len(paths)} documents")
```

Then extend `generate()` by hand until the corpus reaches 60–80 documents, covering both brands, both functions, and all seven doc types, and containing every seeded fact:

| Seeded fact | Documents that must exist |
|---|---|
| `amber_glass_price` | the two Korent quotes above, linked by `supersedes` |
| `korent_moq_contested` | a `spec_sheet` stating MOQ 5,000 and a `contract` annexe stating 8,000, both current, neither superseding the other |
| `korent_cross_brand` | a `grove` vendor quote from Korent at a different unit price and payment term |
| `unanswerable_gap` | **no document** mentions freight insurance excess. Verify with `grep -ri "freight insurance" corpus/out/` returning nothing |
| `mango_variant_archaeology` | a `grove` `decision_memo` citing a superseded consumer-panel `transcript` |

Bulk documents may be LLM-generated, but every file must pass the front-matter validation in Task 8. Add `corpus/README.md` stating in bold that the corpus is synthetic.

- [ ] **Step 6: Run the generator and verify the gap really is a gap**

```bash
python corpus/generate.py
ls corpus/out | wc -l          # expect 60-80
grep -ri "freight insurance" corpus/out/ ; echo "exit=$?"   # expect exit=1, no matches
```

- [ ] **Step 7: Commit**

```bash
git add corpus/ backend/tests/test_corpus_seeds.py backend/pyproject.toml
git commit -m "Add synthetic corpus generator and the five seeded facts"
```

---

## Task 7: Drive connector

**Files:**
- Create: `backend/think9/ingest/drive.py`
- Test: `backend/tests/test_drive.py`

**Interfaces:**
- Consumes: `Settings` (Task 1).
- Produces: `DriveClient(service)` with `list_folder(folder_id) -> list[DriveFile]` and `fetch(file: DriveFile) -> bytes`; `build_service(credentials_json: str)`; `DriveFile` dataclass with `id`, `name`, `mime_type`, `modified_time`, `web_view_link`.

Google Docs are exported as `text/plain`; everything else is downloaded with `get_media`. Tests use a fake service object — no network in the suite.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_drive.py
import pytest

from think9.ingest.drive import DriveClient, DriveFile


class FakeExecutable:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class FakeFiles:
    def __init__(self, listing):
        self.listing = listing
        self.calls: list[dict] = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return FakeExecutable(self.listing)

    def export_media(self, fileId, mimeType):
        self.calls.append({"export": fileId, "mimeType": mimeType})
        return FakeExecutable(b"exported text")

    def get_media(self, fileId):
        self.calls.append({"get_media": fileId})
        return FakeExecutable(b"%PDF-1.4 binary")


class FakeService:
    def __init__(self, listing):
        self._files = FakeFiles(listing)

    def files(self):
        return self._files


def test_list_folder_filters_on_parent_and_excludes_trashed():
    service = FakeService({"files": [
        {"id": "f1", "name": "korent-quote-2026-01", "mimeType": "application/vnd.google-apps.document",
         "modifiedTime": "2026-01-08T10:00:00.000Z", "webViewLink": "https://drive/f1"},
    ]})
    client = DriveClient(service)

    files = client.list_folder("folder-abc")

    assert files == [DriveFile("f1", "korent-quote-2026-01",
                               "application/vnd.google-apps.document",
                               "2026-01-08T10:00:00.000Z", "https://drive/f1")]
    query = service.files().calls[0]["q"]
    assert "'folder-abc' in parents" in query and "trashed = false" in query


def test_google_docs_are_exported_as_plain_text():
    service = FakeService({"files": []})
    client = DriveClient(service)
    doc = DriveFile("f1", "memo", "application/vnd.google-apps.document", "t", "link")

    assert client.fetch(doc) == b"exported text"
    assert service.files().calls[-1]["mimeType"] == "text/plain"


def test_binary_files_are_downloaded_not_exported():
    service = FakeService({"files": []})
    client = DriveClient(service)
    pdf = DriveFile("f2", "contract", "application/pdf", "t", "link")

    assert client.fetch(pdf) == b"%PDF-1.4 binary"
    assert "get_media" in service.files().calls[-1]


def test_pagination_follows_the_next_page_token():
    class Paged(FakeFiles):
        def list(self, **kwargs):
            self.calls.append(kwargs)
            if "pageToken" not in kwargs:
                return FakeExecutable({"files": [_raw("f1")], "nextPageToken": "page-2"})
            return FakeExecutable({"files": [_raw("f2")]})

    def _raw(fid):
        return {"id": fid, "name": fid, "mimeType": "text/plain",
                "modifiedTime": "t", "webViewLink": "link"}

    service = FakeService({"files": []})
    service._files = Paged({"files": []})
    client = DriveClient(service)

    assert [f.id for f in client.list_folder("folder-abc")] == ["f1", "f2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_drive.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'think9.ingest.drive'`

- [ ] **Step 3: Write drive.py**

```python
# backend/think9/ingest/drive.py
import json
from dataclasses import dataclass

from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
GOOGLE_DOC = "application/vnd.google-apps.document"
GOOGLE_SHEET = "application/vnd.google-apps.spreadsheet"
_FIELDS = "nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink)"


@dataclass(frozen=True)
class DriveFile:
    id: str
    name: str
    mime_type: str
    modified_time: str
    web_view_link: str


def build_service(credentials_json: str):
    info = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


class DriveClient:
    def __init__(self, service) -> None:
        self.service = service

    def list_folder(self, folder_id: str) -> list[DriveFile]:
        files: list[DriveFile] = []
        page_token: str | None = None
        while True:
            kwargs = {
                "q": f"'{folder_id}' in parents and trashed = false",
                "fields": _FIELDS,
                "pageSize": 100,
            }
            if page_token:
                kwargs["pageToken"] = page_token
            response = self.service.files().list(**kwargs).execute()
            files.extend(
                DriveFile(f["id"], f["name"], f["mimeType"], f["modifiedTime"], f["webViewLink"])
                for f in response.get("files", [])
            )
            page_token = response.get("nextPageToken")
            if not page_token:
                return files

    def fetch(self, file: DriveFile) -> bytes:
        if file.mime_type in (GOOGLE_DOC, GOOGLE_SHEET):
            mime = "text/csv" if file.mime_type == GOOGLE_SHEET else "text/plain"
            return self.service.files().export_media(fileId=file.id, mimeType=mime).execute()
        return self.service.files().get_media(fileId=file.id).execute()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_drive.py -v`
Expected: 4 passed

- [ ] **Step 5: Wire the real credentials and prove it against live Drive**

Create a GCP project, enable the Drive API, create a service account, download its JSON key, and share the `Think9 Brain — Synthetic Corpus` Drive folder with the service account's email as Viewer. Then:

```bash
cd backend && GOOGLE_CREDENTIALS_JSON="$(cat ../service-account.json)" DRIVE_FOLDER_ID=... \
  uv run python -c "
from think9.config import get_settings
from think9.ingest.drive import DriveClient, build_service
s = get_settings()
files = DriveClient(build_service(s.google_credentials_json)).list_folder(s.drive_folder_id)
print(len(files), files[0].name if files else '')"
```

Expected: the file count matches `ls corpus/out | wc -l`.

**If this stalls past half a day**, take the §12 fallback: add `LocalFolderClient` to `drive.py` with the identical `list_folder`/`fetch` interface reading `corpus/out/`, select it in Task 8 when `GOOGLE_CREDENTIALS_JSON` is unset, and say so plainly in the README. Do not fake Drive.

- [ ] **Step 6: Commit**

```bash
git add backend/think9/ingest/drive.py backend/tests/test_drive.py
git commit -m "Add Drive v3 service-account connector"
```

---

## Task 8: Ingest pipeline

**Files:**
- Create: `backend/think9/ingest/pipeline.py`
- Test: `backend/tests/test_ingest_pipeline.py`

**Interfaces:**
- Consumes: `DriveClient` (Task 7), `parse`/`parse_pdf` (Task 4), `Embedder` (Task 5), `Repository` (Task 3).
- Produces: `parse_front_matter(text) -> tuple[dict, str]`; `to_document(file: DriveFile, meta: dict, body: str) -> Document`; `ingest(client, repo, embedder, folder_id) -> IngestReport`. `IngestReport` fields: `ingested: int`, `skipped_unchanged: int`, `failures: list[tuple[str, str]]`.

Required provenance fields are `brand_id`, `function`, `doc_type`, `effective_date`, `acl`. A document missing any of them raises `MissingProvenance` and is recorded in `failures` — it never enters the index degraded. `supersedes` is legitimately absent for most documents.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_ingest_pipeline.py
from datetime import date

import pytest

from think9.ingest.pipeline import MissingProvenance, parse_front_matter, to_document
from think9.ingest.drive import DriveFile

FILE = DriveFile("f1", "korent-quote-2026-01", "application/vnd.google-apps.document",
                 "2026-01-08T10:00:00.000Z", "https://drive/f1")

DOC = """---
brand_id: nuvia
function: procurement
doc_type: vendor_quote
author: arun@think9.test
effective_date: 2026-01-08
supersedes: null
acl: [procurement]
sensitive: false
---
# Korent Quote

## Pricing
Rs 22.10 per unit.
"""


def test_front_matter_is_split_from_the_body():
    meta, body = parse_front_matter(DOC)

    assert meta["brand_id"] == "nuvia"
    assert meta["acl"] == ["procurement"]
    assert meta["sensitive"] is False
    assert body.startswith("# Korent Quote")
    assert "brand_id" not in body


def test_to_document_carries_provenance_and_the_deep_link():
    meta, body = parse_front_matter(DOC)

    doc = to_document(FILE, meta, body)

    assert doc.deep_link == "https://drive/f1"
    assert doc.effective_date == date(2026, 1, 8)
    assert doc.acl == ("procurement",)
    assert doc.supersedes_id is None
    assert doc.content_hash != ""


@pytest.mark.parametrize("missing", ["brand_id", "function", "doc_type", "effective_date", "acl"])
def test_a_document_missing_required_provenance_fails_loudly(missing):
    meta, body = parse_front_matter(DOC)
    del meta[missing]

    with pytest.raises(MissingProvenance, match=missing):
        to_document(FILE, meta, body)


def test_absent_supersedes_is_not_an_error():
    meta, body = parse_front_matter(DOC)
    del meta["supersedes"]

    assert to_document(FILE, meta, body).supersedes_id is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_ingest_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'think9.ingest.pipeline'`

- [ ] **Step 3: Write pipeline.py**

```python
# backend/think9/ingest/pipeline.py
import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from think9.ingest.drive import GOOGLE_DOC, DriveClient, DriveFile
from think9.ingest.loaders import parse, parse_pdf
from think9.models import Document
from think9.retrieval.embed import Embedder
from think9.store.repository import Repository

REQUIRED = ("brand_id", "function", "doc_type", "effective_date", "acl")
_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


class MissingProvenance(ValueError):
    pass


@dataclass
class IngestReport:
    ingested: int = 0
    skipped_unchanged: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)


def parse_front_matter(text: str) -> tuple[dict, str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    meta: dict = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, raw = line.partition(":")
        meta[key.strip()] = _coerce(raw.strip())
    return meta, text[match.end():]


def _coerce(raw: str):
    if raw in ("null", ""):
        return None
    if raw in ("true", "false"):
        return raw == "true"
    if raw.startswith("[") and raw.endswith("]"):
        return [item.strip() for item in raw[1:-1].split(",") if item.strip()]
    return raw


def _document_uuid(source_id: str) -> "uuid5":
    return uuid5(NAMESPACE_URL, f"google_drive:{source_id}")


def to_document(file: DriveFile, meta: dict, body: str) -> Document:
    for key in REQUIRED:
        if meta.get(key) in (None, [], ""):
            raise MissingProvenance(
                f"{file.name}: required provenance field {key!r} is missing. "
                "Provenance is what makes a citation clickable and an ACL enforceable."
            )
    supersedes = meta.get("supersedes")
    return Document(
        id=_document_uuid(file.id),
        source_system="google_drive",
        source_id=file.id,
        deep_link=file.web_view_link,
        title=file.name,
        doc_type=meta["doc_type"],
        brand_id=meta["brand_id"],
        function=meta["function"],
        author=meta.get("author", "unknown"),
        created_at=datetime.now(timezone.utc),
        effective_date=date.fromisoformat(str(meta["effective_date"])),
        supersedes_id=_document_uuid(supersedes) if supersedes else None,
        acl=tuple(meta["acl"]),
        sensitive=bool(meta.get("sensitive", False)),
        content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
    )


def ingest(client: DriveClient, repo: Repository, embedder: Embedder, folder_id: str) -> IngestReport:
    report = IngestReport()
    for file in client.list_folder(folder_id):
        try:
            raw = client.fetch(file)
            if file.mime_type == "application/pdf":
                meta, chunks = {}, parse_pdf(raw)
                body = "\n".join(c.text for c in chunks)
            else:
                meta, body = parse_front_matter(raw.decode("utf-8"))
                chunks = parse(body, meta.get("doc_type", "transcript"))
            document = to_document(file, meta, body)
            existing = repo.get_document(document.id)
            if existing and existing.content_hash == document.content_hash:
                report.skipped_unchanged += 1
                continue
            repo.upsert_document(document)
            repo.insert_chunks(document.id, chunks, embedder.embed_chunks(chunks))
            report.ingested += 1
        except Exception as exc:  # recorded, never silently swallowed
            report.failures.append((file.name, str(exc)))
    return report
```

**Note on `supersedes`:** the front matter names the superseding target by *filename*, but `_document_uuid` keys on the Drive *file id*. Resolve this by making the generator emit the Drive file id after upload, or by running a second pass that maps titles to ids. Implement the second pass now: after the loop, re-read documents whose `supersedes` was a filename and patch `supersedes_id` from the title index. Add a test asserting the two Korent quotes end up linked.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_ingest_pipeline.py -v`
Expected: 7 passed (4 named + 3 parametrized cases beyond the first)

- [ ] **Step 5: Ingest the real corpus end to end**

```bash
cd backend && uv run python -m think9.ingest.pipeline   # add a __main__ block that wires config
psql "$DATABASE_URL" -c "SELECT count(*) FROM documents; SELECT count(*) FROM chunks;"
```

Expected: document count matches the corpus, chunk count 700–1,200, `failures` empty.

- [ ] **Step 6: Commit**

```bash
git add backend/think9/ingest/pipeline.py backend/tests/test_ingest_pipeline.py
git commit -m "Add ingest pipeline with required-provenance validation"
```

---

## Task 9: Dense and sparse search, both ACL-filtered

**Files:**
- Create: `backend/think9/retrieval/search.py`
- Test: `backend/tests/test_search.py`

**Interfaces:**
- Consumes: `Repository`/connection (Task 3), `Candidate` (Task 2), `Embedder` (Task 5).
- Produces: `dense_search(conn, query_vector, user_groups, limit=30) -> list[Candidate]`, `sparse_search(conn, question, user_groups, limit=30) -> list[Candidate]`.

Access control is enforced here, in SQL, not after generation. Filtering after generation leaks; filtering before retrieval cannot.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_search.py
from tests.conftest import embedding
from tests.test_repository import make_document
from think9.models import ParsedChunk
from think9.retrieval.search import dense_search, sparse_search
from think9.store.repository import Repository


def _seed(conn, *, acl, text, vector):
    repo = Repository(conn)
    doc = make_document(acl=acl)
    repo.upsert_document(doc)
    repo.insert_chunks(doc.id, [ParsedChunk(0, "Pricing", text)], [vector])
    return doc


def test_dense_search_returns_ranked_candidates(conn):
    _seed(conn, acl=("procurement",), text="amber glass Rs 22.10", vector=embedding(0.9))
    _seed(conn, acl=("procurement",), text="annual leave policy", vector=embedding(0.1))

    results = dense_search(conn, embedding(0.9), user_groups=["procurement"])

    assert results[0].text == "amber glass Rs 22.10"
    assert results[0].rank == 1
    assert results[0].source == "dense"


def test_dense_search_hides_chunks_the_user_cannot_open(conn):
    _seed(conn, acl=("legal",), text="settlement terms", vector=embedding(0.9))

    assert dense_search(conn, embedding(0.9), user_groups=["procurement"]) == []


def test_sparse_search_finds_an_exact_entity_token(conn):
    _seed(conn, acl=("procurement",), text="Korent Glassworks SKU AMB-50-FL", vector=embedding(0.1))
    _seed(conn, acl=("procurement",), text="general packaging notes", vector=embedding(0.1))

    results = sparse_search(conn, "AMB-50-FL", user_groups=["procurement"])

    assert len(results) == 1
    assert "AMB-50-FL" in results[0].text
    assert results[0].source == "sparse"


def test_sparse_search_also_enforces_acl(conn):
    _seed(conn, acl=("legal",), text="Korent indemnity clause 7.3", vector=embedding(0.1))

    assert sparse_search(conn, "indemnity clause", user_groups=["procurement"]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_search.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'think9.retrieval.search'`

- [ ] **Step 3: Write search.py**

```python
# backend/think9/retrieval/search.py
import psycopg
from pgvector import Vector

from think9.models import Candidate

_SELECT = """SELECT c.id, c.document_id, c.text, c.heading_path, {score} AS score
             FROM chunks c JOIN documents d ON d.id = c.document_id
             WHERE d.acl && %(groups)s::text[]"""


def dense_search(
    conn: psycopg.Connection, query_vector: list[float], user_groups: list[str], limit: int = 30
) -> list[Candidate]:
    sql = _SELECT.format(score="1 - (c.embedding <=> %(vec)s)") + \
        " ORDER BY c.embedding <=> %(vec)s LIMIT %(limit)s"
    rows = conn.execute(
        sql, {"vec": Vector(query_vector), "groups": user_groups, "limit": limit}
    ).fetchall()
    return _to_candidates(rows, "dense")


def sparse_search(
    conn: psycopg.Connection, question: str, user_groups: list[str], limit: int = 30
) -> list[Candidate]:
    sql = _SELECT.format(score="ts_rank_cd(c.tsv, q)") + \
        " AND c.tsv @@ q ORDER BY ts_rank_cd(c.tsv, q) DESC LIMIT %(limit)s"
    sql = sql.replace("FROM chunks c", "FROM chunks c, websearch_to_tsquery('english', %(q)s) q")
    rows = conn.execute(
        sql, {"q": question, "groups": user_groups, "limit": limit}
    ).fetchall()
    return _to_candidates(rows, "sparse")


def _to_candidates(rows: list[tuple], source: str) -> list[Candidate]:
    return [
        Candidate(chunk_id=r[0], document_id=r[1], text=r[2], heading_path=r[3],
                  score=float(r[4]), rank=i + 1, source=source)
        for i, r in enumerate(rows)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_search.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add backend/think9/retrieval/search.py backend/tests/test_search.py
git commit -m "Add ACL-filtered dense and sparse search"
```

---

## Task 10: Reciprocal rank fusion and cross-encoder rerank

**Files:**
- Create: `backend/think9/retrieval/fusion.py`, `backend/think9/retrieval/rerank.py`
- Test: `backend/tests/test_fusion.py`, `backend/tests/test_rerank.py`

**Interfaces:**
- Consumes: `Candidate` (Task 2).
- Produces: `reciprocal_rank_fusion(rankings: list[list[Candidate]], k: int = 60) -> list[Candidate]` (returns `source="fused"`); `Reranker` with `rerank(question, candidates, top_n=8) -> list[Candidate]` (returns `source="reranked"`).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_fusion.py
from uuid import uuid4

from think9.models import Candidate
from think9.retrieval.fusion import reciprocal_rank_fusion

A, B, C = uuid4(), uuid4(), uuid4()


def _c(chunk_id, rank, source):
    return Candidate(chunk_id=chunk_id, document_id=uuid4(), text=str(chunk_id),
                     heading_path="h", score=1.0 / rank, rank=rank, source=source)


def test_a_chunk_ranked_by_both_arms_beats_one_ranked_by_only_one():
    dense = [_c(A, 1, "dense"), _c(B, 2, "dense")]
    sparse = [_c(B, 1, "sparse"), _c(C, 2, "sparse")]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert fused[0].chunk_id == B


def test_fusion_scores_follow_the_rrf_formula():
    fused = reciprocal_rank_fusion([[_c(A, 1, "dense")]], k=60)
    assert fused[0].score == 1 / 61


def test_ranks_are_renumbered_from_one_and_marked_fused():
    fused = reciprocal_rank_fusion([[_c(A, 1, "dense"), _c(B, 2, "dense")]])
    assert [c.rank for c in fused] == [1, 2]
    assert all(c.source == "fused" for c in fused)


def test_empty_rankings_produce_no_candidates():
    assert reciprocal_rank_fusion([[], []]) == []
```

```python
# backend/tests/test_rerank.py
from uuid import uuid4

from think9.models import Candidate
from think9.retrieval.rerank import Reranker


def _c(text, rank):
    return Candidate(chunk_id=uuid4(), document_id=uuid4(), text=text,
                     heading_path="h", score=0.5, rank=rank, source="fused")


def test_reranker_promotes_the_genuinely_relevant_passage():
    candidates = [
        _c("Employees accrue 18 days of annual leave per year.", 1),
        _c("50ml amber glass jars are priced at Rs 22.10 per unit.", 2),
    ]

    reranked = Reranker().rerank("what do we pay for amber glass", candidates)

    assert "Rs 22.10" in reranked[0].text
    assert reranked[0].rank == 1
    assert reranked[0].source == "reranked"


def test_reranker_truncates_to_top_n():
    candidates = [_c(f"passage {i}", i + 1) for i in range(12)]
    assert len(Reranker().rerank("anything", candidates, top_n=8)) == 8


def test_reranking_an_empty_list_is_not_an_error():
    assert Reranker().rerank("anything", []) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_fusion.py tests/test_rerank.py -v`
Expected: FAIL — both modules missing

- [ ] **Step 3: Write fusion.py**

```python
# backend/think9/retrieval/fusion.py
"""Reciprocal rank fusion.

Fuses ranked lists without needing their scores to be comparable, which matters because
a cosine similarity and a ts_rank_cd are not on the same scale and never will be.
"""
from collections import defaultdict

from think9.models import Candidate


def reciprocal_rank_fusion(rankings: list[list[Candidate]], k: int = 60) -> list[Candidate]:
    scores: dict = defaultdict(float)
    representative: dict = {}
    for ranking in rankings:
        for candidate in ranking:
            scores[candidate.chunk_id] += 1.0 / (k + candidate.rank)
            representative.setdefault(candidate.chunk_id, candidate)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[Candidate] = []
    for position, (chunk_id, score) in enumerate(ordered, start=1):
        base = representative[chunk_id]
        fused.append(
            Candidate(chunk_id=base.chunk_id, document_id=base.document_id, text=base.text,
                      heading_path=base.heading_path, score=score, rank=position, source="fused")
        )
    return fused
```

- [ ] **Step 4: Write rerank.py**

```python
# backend/think9/retrieval/rerank.py
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from think9.config import get_settings
from think9.models import Candidate


@lru_cache(maxsize=1)
def _encoder() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=get_settings().reranker_model)


class Reranker:
    def rerank(self, question: str, candidates: list[Candidate], top_n: int = 8) -> list[Candidate]:
        if not candidates:
            return []
        scores = list(_encoder().rerank(question, [c.text for c in candidates]))
        ordered = sorted(zip(candidates, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [
            Candidate(chunk_id=c.chunk_id, document_id=c.document_id, text=c.text,
                      heading_path=c.heading_path, score=float(score), rank=position,
                      source="reranked")
            for position, (c, score) in enumerate(ordered[:top_n], start=1)
        ]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_fusion.py tests/test_rerank.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add backend/think9/retrieval/fusion.py backend/think9/retrieval/rerank.py backend/tests/test_fusion.py backend/tests/test_rerank.py
git commit -m "Add reciprocal rank fusion and cross-encoder reranking"
```

---

## Task 11: Temporal authority layer

**Files:**
- Create: `backend/think9/retrieval/temporal.py`
- Test: `backend/tests/test_temporal.py`

**Interfaces:**
- Consumes: `Document`, `RetrievedChunk`, `Route` (Task 2).
- Produces: `apply_temporal_authority(chunks: list[RetrievedChunk], route: Route) -> list[RetrievedChunk]`, `as_of_date(chunks) -> date | None`.

This is the layer most retrieval systems omit and the one that decides whether an operations team trusts the thing after week two. It is **route-aware**: for `decision_archaeology`, superseded documents are precisely what the question is about, so demotion is disabled (§4.4).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_temporal.py
from datetime import date

from tests.test_repository import make_document
from think9.models import RetrievedChunk
from think9.retrieval.temporal import apply_temporal_authority, as_of_date

OLD = make_document(title="Korent Quote 2024", effective_date=date(2024, 3, 12))
NEW = make_document(title="Korent Quote 2026", effective_date=date(2026, 1, 8),
                    supersedes_id=OLD.id)


def _chunk(doc, text, score):
    return RetrievedChunk(chunk_id=doc.id, document=doc, heading_path="Pricing",
                          text=text, score=score)


def test_a_superseded_document_is_demoted_below_its_successor():
    chunks = [_chunk(OLD, "Rs 18.40 per unit", 0.95), _chunk(NEW, "Rs 22.10 per unit", 0.60)]

    result = apply_temporal_authority(chunks, route="factual_lookup")

    assert result[0].document.id == NEW.id
    assert result[1].demoted is True
    assert result[1].demoted_by == NEW.id


def test_demotion_is_disabled_for_decision_archaeology():
    chunks = [_chunk(OLD, "Rs 18.40 per unit", 0.95), _chunk(NEW, "Rs 22.10 per unit", 0.60)]

    result = apply_temporal_authority(chunks, route="decision_archaeology")

    assert result[0].document.id == OLD.id
    assert all(c.demoted is False for c in result)


def test_a_superseded_document_absent_its_successor_is_still_demoted():
    result = apply_temporal_authority([_chunk(OLD, "Rs 18.40", 0.9)], route="factual_lookup")
    assert result[0].demoted is True


def test_unrelated_documents_are_left_alone():
    other = make_document(title="Leave policy", effective_date=date(2025, 6, 1))
    result = apply_temporal_authority([_chunk(other, "18 days", 0.7)], route="policy")
    assert result[0].demoted is False


def test_as_of_is_the_latest_effective_date_among_undemoted_chunks():
    chunks = apply_temporal_authority(
        [_chunk(OLD, "Rs 18.40", 0.95), _chunk(NEW, "Rs 22.10", 0.60)], route="factual_lookup"
    )
    assert as_of_date(chunks) == date(2026, 1, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_temporal.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'think9.retrieval.temporal'`

- [ ] **Step 3: Write temporal.py**

```python
# backend/think9/retrieval/temporal.py
"""Business facts decay. A vendor price from 2024 is not the current price.

Without this layer the system confidently quotes a dead price with a perfect citation,
which is exactly the failure that destroys trust.
"""
from dataclasses import replace
from datetime import date

from think9.models import RetrievedChunk, Route

DEMOTION_PENALTY = 0.5


def apply_temporal_authority(chunks: list[RetrievedChunk], route: Route) -> list[RetrievedChunk]:
    if route == "decision_archaeology":
        # History is what the question is about. Demoting it would break the query type.
        return sorted(chunks, key=lambda c: c.score, reverse=True)

    superseded_ids = {
        c.document.supersedes_id: c.document.id
        for c in chunks
        if c.document.supersedes_id is not None
    }

    judged: list[RetrievedChunk] = []
    for chunk in chunks:
        doc = chunk.document
        successor = superseded_ids.get(doc.id)
        is_superseded = successor is not None or _has_successor_flag(doc)
        if is_superseded:
            judged.append(replace(chunk, score=chunk.score * DEMOTION_PENALTY,
                                  demoted=True, demoted_by=successor))
        else:
            judged.append(chunk)
    return sorted(judged, key=lambda c: (not c.demoted, c.score), reverse=True)


def _has_successor_flag(doc) -> bool:
    """A document known to be superseded even when its successor was not retrieved.

    Set at ingestion: the second pass in Task 8 records the reverse link, so a lineage
    head is identifiable without needing both ends in the result set.
    """
    return getattr(doc, "is_superseded", False)


def as_of_date(chunks: list[RetrievedChunk]) -> date | None:
    live = [c.document.effective_date for c in chunks if not c.demoted]
    return max(live) if live else None
```

**Note:** `_has_successor_flag` needs a real backing field. Add `is_superseded: bool = False` to `Document` in `models.py`, populate it in Task 8's second pass (`UPDATE documents SET is_superseded = true WHERE id IN (SELECT supersedes_id FROM documents WHERE supersedes_id IS NOT NULL)`), add the column to `schema.sql`, and update `_row_to_document`. Test `test_a_superseded_document_absent_its_successor_is_still_demoted` constructs `OLD` with `is_superseded=True`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_temporal.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add backend/think9/retrieval/temporal.py backend/think9/models.py backend/think9/store/ backend/tests/test_temporal.py
git commit -m "Add route-aware temporal authority layer"
```

---

## Task 12: Retriever composition

**Files:**
- Create: `backend/think9/retrieval/retriever.py`
- Test: `backend/tests/test_retriever.py`

**Interfaces:**
- Consumes: everything in `retrieval/` plus `Repository`.
- Produces: `Retriever(conn, embedder, reranker)` with `retrieve(question, route, user_groups, use_hybrid=True, use_rerank=True, use_temporal=True) -> RetrievalResult`. `RetrievalResult` fields: `chunks: list[RetrievedChunk]`, `as_of: date | None`, `coverage: float`, `trace: dict`.

The three boolean flags exist so Task 20's ablations toggle real code paths rather than reimplementing the pipeline.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_retriever.py
from tests.conftest import embedding
from tests.test_repository import make_document
from think9.models import ParsedChunk
from think9.retrieval.embed import Embedder
from think9.retrieval.rerank import Reranker
from think9.retrieval.retriever import Retriever
from think9.store.repository import Repository


def _seed(conn, embedder, doc, text, heading="Pricing"):
    repo = Repository(conn)
    repo.upsert_document(doc)
    chunk = ParsedChunk(0, heading, text)
    repo.insert_chunks(doc.id, [chunk], embedder.embed_chunks([chunk]))


def test_retrieval_returns_chunks_with_coverage_and_an_as_of_date(conn):
    embedder = Embedder()
    _seed(conn, embedder, make_document(), "50ml amber glass is Rs 22.10 per unit")
    retriever = Retriever(conn, embedder, Reranker())

    result = retriever.retrieve("what do we pay for amber glass", "factual_lookup", ["procurement"])

    assert result.chunks
    assert result.coverage > 0
    assert result.as_of is not None


def test_trace_records_every_stage(conn):
    embedder = Embedder()
    _seed(conn, embedder, make_document(), "50ml amber glass is Rs 22.10 per unit")
    retriever = Retriever(conn, embedder, Reranker())

    trace = retriever.retrieve("amber glass", "factual_lookup", ["procurement"]).trace

    assert set(trace) >= {"dense", "sparse", "fused", "reranked", "demoted"}


def test_disabling_hybrid_skips_the_sparse_arm(conn):
    embedder = Embedder()
    _seed(conn, embedder, make_document(), "50ml amber glass is Rs 22.10 per unit")
    retriever = Retriever(conn, embedder, Reranker())

    trace = retriever.retrieve("amber glass", "factual_lookup", ["procurement"],
                               use_hybrid=False).trace

    assert trace["sparse"] == []


def test_coverage_is_zero_when_nothing_is_retrievable(conn):
    retriever = Retriever(conn, Embedder(), Reranker())

    result = retriever.retrieve("freight insurance excess", "factual_lookup", ["procurement"])

    assert result.chunks == []
    assert result.coverage == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_retriever.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'think9.retrieval.retriever'`

- [ ] **Step 3: Write retriever.py**

```python
# backend/think9/retrieval/retriever.py
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import psycopg

from think9.models import Candidate, RetrievedChunk, Route
from think9.retrieval.embed import Embedder
from think9.retrieval.fusion import reciprocal_rank_fusion
from think9.retrieval.rerank import Reranker
from think9.retrieval.search import dense_search, sparse_search
from think9.retrieval.temporal import apply_temporal_authority, as_of_date
from think9.store.repository import Repository


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    as_of: date | None
    coverage: float
    trace: dict[str, Any] = field(default_factory=dict)


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
        shortlist = self.reranker.rerank(question, fused) if use_rerank else fused[:8]

        enriched = [self._enrich(c) for c in shortlist]
        enriched = [c for c in enriched if c is not None]
        judged = apply_temporal_authority(enriched, route) if use_temporal else enriched

        trace = {
            "dense": _summarise(dense),
            "sparse": _summarise(sparse),
            "fused": _summarise(fused),
            "reranked": _summarise(shortlist),
            "demoted": [
                {"title": c.document.title, "demoted_by": str(c.demoted_by)}
                for c in judged if c.demoted
            ],
        }
        return RetrievalResult(
            chunks=judged,
            as_of=as_of_date(judged),
            coverage=judged[0].score if judged and not judged[0].demoted else 0.0,
            trace=trace,
        )

    def _enrich(self, candidate: Candidate) -> RetrievedChunk | None:
        document = self.repo.get_document(candidate.document_id)
        if document is None:
            return None
        return RetrievedChunk(
            chunk_id=candidate.chunk_id, document=document,
            heading_path=candidate.heading_path, text=candidate.text, score=candidate.score,
        )


def _renumber(candidates: list[Candidate]) -> list[Candidate]:
    from dataclasses import replace
    return [replace(c, rank=i + 1, source="fused") for i, c in enumerate(candidates)]


def _summarise(candidates: list[Candidate]) -> list[dict]:
    return [
        {"chunk_id": str(c.chunk_id), "rank": c.rank, "score": round(c.score, 4),
         "snippet": c.text[:120]}
        for c in candidates[:10]
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_retriever.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add backend/think9/retrieval/retriever.py backend/tests/test_retriever.py
git commit -m "Compose the full retrieval pipeline behind one interface"
```

---

## Task 13: LLM client and router

**Files:**
- Create: `backend/think9/agent/__init__.py`, `backend/think9/agent/llm.py`, `backend/think9/agent/router.py`
- Test: `backend/tests/test_router.py`

**Interfaces:**
- Consumes: `Settings` (Task 1), `Route` (Task 2).
- Produces: `LLM` with `complete(system: str, user: str, model: str | None = None) -> str`; `classify(question: str, llm: LLM | None = None) -> Route`; `classify_deterministic(question: str) -> Route`.

Routing every query through the large model is the most common and most avoidable cost sink — hence a small model here, with a deterministic classifier as the fallback when no key is configured or the model returns something unrecognised.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_router.py
import pytest

from think9.agent.router import classify, classify_deterministic


class StubLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, model: str | None = None) -> str:
        self.calls.append((system, user))
        return self.reply


@pytest.mark.parametrize("question,expected", [
    ("What do we pay for 50ml amber glass?", "factual_lookup"),
    ("Which brands buy from Korent and on what terms?", "cross_brand_comparison"),
    ("Why did we discontinue the mango variant?", "decision_archaeology"),
    ("What is our standard exclusivity clause?", "policy"),
    ("Show me total spend by vendor last quarter", "needs_structured_data"),
])
def test_deterministic_classifier_covers_all_five_routes(question, expected):
    assert classify_deterministic(question) == expected


def test_model_classification_is_used_when_it_returns_a_known_route():
    llm = StubLLM("cross_brand_comparison")
    assert classify("anything at all", llm) == "cross_brand_comparison"
    assert llm.calls


def test_unrecognised_model_output_falls_back_to_the_deterministic_classifier():
    assert classify("Why did we kill the mango variant?", StubLLM("banana")) == \
        "decision_archaeology"


def test_no_llm_means_deterministic_only():
    assert classify("What do we pay for amber glass?", None) == "factual_lookup"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_router.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'think9.agent.router'`

- [ ] **Step 3: Write llm.py**

```python
# backend/think9/agent/llm.py
from openai import OpenAI

from think9.config import get_settings


class LLM:
    """The only place that talks to a model. OpenAI-compatible, so it points at Groq."""

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

    def complete(self, system: str, user: str, model: str | None = None) -> str:
        response = self._client.chat.completions.create(
            model=model or self._settings.llm_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
        )
        return (response.choices[0].message.content or "").strip()
```

- [ ] **Step 4: Write router.py**

```python
# backend/think9/agent/router.py
import re

from think9.config import get_settings
from think9.models import Route

ROUTES: tuple[Route, ...] = (
    "factual_lookup", "cross_brand_comparison", "policy",
    "decision_archaeology", "needs_structured_data",
)

_SYSTEM = (
    "Classify the operations question into exactly one of: "
    + ", ".join(ROUTES)
    + ". Reply with the label only, no punctuation or explanation."
)

_ARCHAEOLOGY = re.compile(r"\bwhy (did|do|was|were|are) we\b|\bwhy was\b|\bdiscontinu|\bkill(ed)?\b", re.I)
_STRUCTURED = re.compile(r"\btotal\b|\bsum\b|\bhow much did we spend\b|\bby vendor\b|\blast quarter\b", re.I)
_CROSS_BRAND = re.compile(r"\bwhich brands?\b|\bacross brands?\b|\bcompare\b|\bboth brands\b", re.I)
_POLICY = re.compile(r"\bpolicy\b|\bstandard\b|\bclause\b|\bwhat is our\b", re.I)


def classify_deterministic(question: str) -> Route:
    if _ARCHAEOLOGY.search(question):
        return "decision_archaeology"
    if _STRUCTURED.search(question):
        return "needs_structured_data"
    if _CROSS_BRAND.search(question):
        return "cross_brand_comparison"
    if _POLICY.search(question):
        return "policy"
    return "factual_lookup"


def classify(question: str, llm=None) -> Route:
    if llm is None:
        return classify_deterministic(question)
    try:
        label = llm.complete(_SYSTEM, question, model=get_settings().router_model).strip().lower()
    except Exception:
        return classify_deterministic(question)
    return label if label in ROUTES else classify_deterministic(question)
```

**Ordering note:** `_STRUCTURED` is checked before `_CROSS_BRAND` and `_POLICY` so "total spend by vendor last quarter" routes to `needs_structured_data`; `_ARCHAEOLOGY` is checked first so "why did we…" wins over a `_POLICY` keyword appearing in the same sentence.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_router.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add backend/think9/agent/ backend/tests/test_router.py
git commit -m "Add LLM client and query router with deterministic fallback"
```

---

## Task 14: Synthesiser and verifier

**Files:**
- Create: `backend/think9/agent/nodes.py` (synthesiser only), `backend/think9/agent/verifier.py`
- Test: `backend/tests/test_verifier.py`, `backend/tests/test_synthesiser.py`

**Interfaces:**
- Consumes: `LLM` (Task 13), `RetrievedChunk`, `Citation` (Task 2).
- Produces: `synthesise(llm, question, chunks) -> tuple[str, tuple[Citation, ...]]`; `verify(draft, chunks, llm=None) -> VerificationResult` with fields `text: str`, `stripped: list[str]`, `refused: bool`, `claims: list[ClaimVerdict]`; `ClaimVerdict` fields `claim: str`, `supported: bool`, `reason: str`.

The verifier is a separate stage rather than an instruction inside the synthesis prompt. Asking a model to check its own output in the same pass is not a control; it is a suggestion.

Verification runs deterministic checks first, then entailment:

1. **Citation validity** — every `[c:<chunk_id>]` marker must resolve to a retrieved chunk.
2. **Numeric grounding** — every digit-string in the claim must appear in some retrieved chunk. Carried from Resilience's Gate 3: a price, an MOQ, a lead time and a clause number are all digit-strings, so one check covers all four.
3. **Entailment** — the model is asked whether the retrieved span entails the claim. This is what catches the correctly-sourced but wrongly-combined claim that Resilience's error analysis identified and could not close.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_verifier.py
from datetime import date
from uuid import uuid4

from tests.test_repository import make_document
from think9.agent.verifier import verify
from think9.models import RetrievedChunk

CHUNK_ID = uuid4()
DOC = make_document(effective_date=date(2026, 1, 8))
CHUNKS = [RetrievedChunk(chunk_id=CHUNK_ID, document=DOC, heading_path="Pricing",
                         text="50ml amber glass is Rs 22.10 per unit at 5,000 units.", score=0.9)]


class YesLLM:
    def complete(self, system, user, model=None):
        return "SUPPORTED"


class NoLLM:
    def complete(self, system, user, model=None):
        return "NOT_SUPPORTED"


def test_a_grounded_claim_survives():
    draft = f"Amber glass costs Rs 22.10 per unit [c:{CHUNK_ID}]."

    result = verify(draft, CHUNKS, YesLLM())

    assert result.refused is False
    assert "22.10" in result.text
    assert result.stripped == []


def test_a_fabricated_number_is_stripped_without_any_model_call():
    draft = f"Amber glass costs Rs 31.75 per unit [c:{CHUNK_ID}]."

    result = verify(draft, CHUNKS, llm=None)

    assert "31.75" not in result.text
    assert result.stripped == [draft]
    assert result.refused is True


def test_an_invalid_citation_is_stripped():
    draft = f"Amber glass costs Rs 22.10 per unit [c:{uuid4()}]."

    result = verify(draft, CHUNKS, YesLLM())

    assert result.refused is True
    assert any("citation" in v.reason for v in result.claims)


def test_a_correctly_sourced_but_wrongly_combined_claim_fails_entailment():
    draft = f"Amber glass costs Rs 22.10 per unit at 8,000 units [c:{CHUNK_ID}]."

    result = verify(draft, CHUNKS, NoLLM())

    assert result.refused is True
    assert any(not v.supported and "entail" in v.reason for v in result.claims)


def test_stripping_every_claim_forces_a_refusal():
    draft = f"Rs 99.99 per unit [c:{CHUNK_ID}]. Lead time is 91 days [c:{CHUNK_ID}]."

    result = verify(draft, CHUNKS, YesLLM())

    assert result.refused is True
    assert result.text == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_verifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'think9.agent.verifier'`

- [ ] **Step 3: Write verifier.py**

```python
# backend/think9/agent/verifier.py
"""A separate pass over the draft. Never a clause in the synthesis prompt."""
import re
from dataclasses import dataclass, field

from think9.models import RetrievedChunk

_CITATION = re.compile(r"\[c:([0-9a-fA-F-]{36})\]")
_DIGITS = re.compile(r"\d[\d,.]*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

_ENTAILMENT_SYSTEM = (
    "You check whether a claim is entailed by the evidence. "
    "Reply with exactly SUPPORTED or NOT_SUPPORTED. "
    "A claim that recombines real numbers into a relationship the evidence does not state "
    "is NOT_SUPPORTED."
)


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool
    reason: str


@dataclass
class VerificationResult:
    text: str
    refused: bool
    stripped: list[str] = field(default_factory=list)
    claims: list[ClaimVerdict] = field(default_factory=list)


def verify(draft: str, chunks: list[RetrievedChunk], llm=None) -> VerificationResult:
    valid_ids = {str(c.chunk_id) for c in chunks}
    corpus = " ".join(c.text for c in chunks)

    kept: list[str] = []
    stripped: list[str] = []
    verdicts: list[ClaimVerdict] = []

    for claim in [s.strip() for s in _SENTENCE.split(draft) if s.strip()]:
        verdict = _judge(claim, valid_ids, corpus, llm)
        verdicts.append(verdict)
        (kept if verdict.supported else stripped).append(claim)

    text = " ".join(kept)
    return VerificationResult(
        text=text, refused=not kept, stripped=stripped, claims=verdicts
    )


def _judge(claim: str, valid_ids: set[str], corpus: str, llm) -> ClaimVerdict:
    cited = _CITATION.findall(claim)
    if not cited:
        return ClaimVerdict(claim, False, "no citation")
    if any(cid not in valid_ids for cid in cited):
        return ClaimVerdict(claim, False, "citation does not resolve to a retrieved chunk")

    bare = _CITATION.sub("", claim)
    for number in _DIGITS.findall(bare):
        normalised = number.rstrip(".,")
        if normalised and normalised not in corpus.replace(" ", " "):
            return ClaimVerdict(claim, False, f"ungrounded number {normalised!r}")

    if llm is None:
        return ClaimVerdict(claim, True, "deterministic checks passed; entailment skipped")

    try:
        reply = llm.complete(_ENTAILMENT_SYSTEM, f"EVIDENCE:\n{corpus}\n\nCLAIM:\n{bare}")
    except Exception:
        return ClaimVerdict(claim, False, "entailment check unavailable")
    if reply.strip().upper().startswith("SUPPORTED"):
        return ClaimVerdict(claim, True, "entailed by evidence")
    return ClaimVerdict(claim, False, "evidence does not entail the claim")
```

**On `test_a_fabricated_number_is_stripped_without_any_model_call`:** `llm=None` still runs the deterministic checks, and `31.75` is absent from the corpus, so the claim is stripped before any model is consulted. That is the point — the cheap check runs first.

- [ ] **Step 4: Write the synthesiser and its test**

```python
# backend/tests/test_synthesiser.py
from uuid import uuid4

from tests.test_repository import make_document
from think9.agent.nodes import synthesise
from think9.models import RetrievedChunk

CHUNK_ID = uuid4()
CHUNKS = [RetrievedChunk(chunk_id=CHUNK_ID, document=make_document(), heading_path="Pricing",
                         text="50ml amber glass is Rs 22.10 per unit.", score=0.9)]


class EchoLLM:
    def __init__(self):
        self.user_prompt = ""

    def complete(self, system, user, model=None):
        self.user_prompt = user
        return f"Amber glass costs Rs 22.10 per unit [c:{CHUNK_ID}]."


def test_synthesis_returns_text_and_resolved_citations():
    text, citations = synthesise(EchoLLM(), "what do we pay for amber glass", CHUNKS)

    assert "22.10" in text
    assert len(citations) == 1
    assert citations[0].chunk_id == CHUNK_ID
    assert citations[0].deep_link == CHUNKS[0].document.deep_link


def test_the_prompt_contains_the_chunk_ids_the_model_must_cite():
    llm = EchoLLM()
    synthesise(llm, "q", CHUNKS)
    assert str(CHUNK_ID) in llm.user_prompt


def test_demoted_chunks_are_excluded_from_the_context():
    from dataclasses import replace
    demoted = [replace(CHUNKS[0], demoted=True)]
    llm = EchoLLM()

    synthesise(llm, "q", demoted)

    assert str(CHUNK_ID) not in llm.user_prompt
```

```python
# backend/think9/agent/nodes.py
from think9.models import Citation, RetrievedChunk

_SYSTEM = (
    "You answer operations questions using ONLY the provided context. "
    "Cite every factual claim inline as [c:<chunk_id>] using the ids given. "
    "Never state a number that does not appear in the context. "
    "If the context does not support an answer, say exactly: INSUFFICIENT_EVIDENCE."
)


def synthesise(llm, question: str, chunks: list[RetrievedChunk]) -> tuple[str, tuple[Citation, ...]]:
    live = [c for c in chunks if not c.demoted]
    context = "\n\n".join(
        f"[c:{c.chunk_id}] ({c.document.title} > {c.heading_path}, "
        f"effective {c.document.effective_date})\n{c.text}"
        for c in live
    )
    text = llm.complete(_SYSTEM, f"CONTEXT:\n{context}\n\nQUESTION:\n{question}")
    citations = tuple(
        Citation(chunk_id=c.chunk_id, document_title=c.document.title,
                 heading_path=c.heading_path, deep_link=c.document.deep_link,
                 effective_date=c.document.effective_date)
        for c in live
        if str(c.chunk_id) in text
    )
    return text, citations
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_verifier.py tests/test_synthesiser.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add backend/think9/agent/nodes.py backend/think9/agent/verifier.py backend/tests/test_verifier.py backend/tests/test_synthesiser.py
git commit -m "Add synthesiser and separate verifier stage with entailment"
```

---

## Task 15: Refusal path and owner routing

**Files:**
- Modify: `backend/think9/agent/nodes.py` (append)
- Test: `backend/tests/test_refusal.py`

**Interfaces:**
- Consumes: `Repository.find_owner` (Task 3), `RetrievedChunk`, `Owner`, `Answer` (Task 2).
- Produces: `resolve_owner(repo, brand_id, function) -> Owner | None`; `build_refusal(question, chunks, owner) -> Answer`.

A confident "I don't know, ask Priya, and here is the closest thing I found" beats twenty minutes of searching, and beats a fabricated answer by an infinite margin.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_refusal.py
from tests.test_repository import make_document
from think9.agent.nodes import build_refusal
from think9.models import Owner, RetrievedChunk
from uuid import uuid4

OWNER = Owner("nuvia", "procurement", "Priya Nair", "priya@think9.test")
NEAR = [RetrievedChunk(chunk_id=uuid4(), document=make_document(title="Korent Quote 2026"),
                       heading_path="Terms", text="Net 45 from invoice date.", score=0.31)]


def test_refusal_names_the_owner_and_the_closest_evidence():
    answer = build_refusal("What is our freight insurance excess?", NEAR, OWNER)

    assert answer.outcome == "refused"
    assert "Priya Nair" in answer.text
    assert "Korent Quote 2026" in answer.text
    assert answer.citations == ()


def test_refusal_without_an_owner_still_refuses_cleanly():
    answer = build_refusal("What is our freight insurance excess?", NEAR, None)

    assert answer.outcome == "refused"
    assert "Priya" not in answer.text
    assert "don't have" in answer.text.lower()


def test_refusal_with_no_near_evidence_says_so():
    answer = build_refusal("What is our freight insurance excess?", [], OWNER)

    assert "closest" not in answer.text.lower()
    assert "Priya Nair" in answer.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_refusal.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_refusal'`

- [ ] **Step 3: Append to nodes.py**

```python
def resolve_owner(repo, brand_id: str, function: str):
    return repo.find_owner(brand_id, function) or repo.find_owner("shared", function)


def build_refusal(question: str, chunks: list[RetrievedChunk], owner) -> Answer:
    parts = ["I don't have this in the indexed corpus."]
    if chunks:
        nearest = chunks[0]
        parts.append(
            f"The closest thing I found is “{nearest.document.title} > {nearest.heading_path}”, "
            f"which covers something adjacent but does not answer it."
        )
    if owner is not None:
        parts.append(f"The person who would know is {owner.person_name} ({owner.contact}).")
    return Answer(text=" ".join(parts), outcome="refused", trace={"question": question})
```

Add `Answer` to the imports at the top of `nodes.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_refusal.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add backend/think9/agent/nodes.py backend/tests/test_refusal.py
git commit -m "Add refusal path with nearest-evidence and owner routing"
```

---

## Task 16: LangGraph wiring

**Files:**
- Create: `backend/think9/agent/state.py`, `backend/think9/agent/graph.py`
- Test: `backend/tests/test_graph.py`

**Interfaces:**
- Consumes: everything in `agent/` and `Retriever` (Task 12).
- Produces: `BrainState` TypedDict; `build_graph(retriever, repo, llm) -> CompiledStateGraph`; `ask(graph, question, user_groups, user_id) -> Answer`.

`BrainState` keys: `question: str`, `user_groups: list[str]`, `route: Route`, `retrieval: RetrievalResult | None`, `owner: Owner | None`, `draft: str`, `citations: tuple[Citation, ...]`, `answer: Answer | None`, `trace: Annotated[dict, merge_trace]`.

The graph is a state machine with inspectable transitions, not one prompt. `merge_trace` is the reducer that lets the two retriever nodes write concurrently to the same `trace` key without clobbering each other.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_graph.py
from datetime import date
from uuid import uuid4

from tests.test_repository import make_document
from think9.agent.graph import ask, build_graph
from think9.models import Owner, RetrievedChunk
from think9.retrieval.retriever import RetrievalResult

CHUNK_ID = uuid4()
DOC = make_document(effective_date=date(2026, 1, 8))
GOOD = RetrievalResult(
    chunks=[RetrievedChunk(chunk_id=CHUNK_ID, document=DOC, heading_path="Pricing",
                           text="50ml amber glass is Rs 22.10 per unit.", score=0.82)],
    as_of=date(2026, 1, 8), coverage=0.82, trace={"dense": [], "sparse": []},
)
EMPTY = RetrievalResult(chunks=[], as_of=None, coverage=0.0, trace={})


class StubRetriever:
    def __init__(self, result): self.result = result
    def retrieve(self, question, route, user_groups, **kwargs): return self.result


class StubRepo:
    def find_owner(self, brand_id, function):
        return Owner(brand_id, function, "Priya Nair", "priya@think9.test")


class StubLLM:
    def complete(self, system, user, model=None):
        if "SUPPORTED" in system:
            return "SUPPORTED"
        if "Classify" in system:
            return "factual_lookup"
        return f"Amber glass costs Rs 22.10 per unit [c:{CHUNK_ID}]."


def test_a_supported_question_is_answered_with_citations_and_an_as_of_date():
    graph = build_graph(StubRetriever(GOOD), StubRepo(), StubLLM())

    answer = ask(graph, "what do we pay for amber glass", ["procurement"], "u1")

    assert answer.outcome == "answered"
    assert answer.as_of == date(2026, 1, 8)
    assert len(answer.citations) == 1


def test_below_threshold_coverage_routes_to_refusal_without_calling_the_synthesiser():
    graph = build_graph(StubRetriever(EMPTY), StubRepo(), StubLLM())

    answer = ask(graph, "what is our freight insurance excess", ["procurement"], "u1")

    assert answer.outcome == "refused"
    assert "Priya Nair" in answer.text


def test_needs_structured_data_is_declined_gracefully():
    graph = build_graph(StubRetriever(GOOD), StubRepo(), StubLLM())

    answer = ask(graph, "show me total spend by vendor last quarter", ["procurement"], "u1")

    assert answer.outcome == "refused"
    assert "procurement tables" in answer.text


def test_the_trace_records_the_route_and_every_stage():
    graph = build_graph(StubRetriever(GOOD), StubRepo(), StubLLM())

    answer = ask(graph, "what do we pay for amber glass", ["procurement"], "u1")

    assert answer.trace["route"] == "factual_lookup"
    assert "retrieval" in answer.trace
    assert "verifier" in answer.trace
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'think9.agent.graph'`

- [ ] **Step 3: Write state.py**

```python
# backend/think9/agent/state.py
from typing import Annotated, Any
from typing_extensions import TypedDict

from think9.models import Answer, Citation, Owner, Route


def merge_trace(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Reducer so parallel nodes can both write to `trace` without clobbering."""
    return {**left, **right}


class BrainState(TypedDict, total=False):
    question: str
    user_groups: list[str]
    user_id: str
    route: Route
    retrieval: Any            # RetrievalResult; Any avoids a circular import
    owner: Owner | None
    draft: str
    citations: tuple[Citation, ...]
    answer: Answer | None
    trace: Annotated[dict[str, Any], merge_trace]
```

- [ ] **Step 4: Write graph.py**

```python
# backend/think9/agent/graph.py
from langgraph.graph import END, START, StateGraph

from think9.agent.nodes import build_refusal, resolve_owner, synthesise
from think9.agent.router import classify
from think9.agent.state import BrainState
from think9.agent.verifier import verify
from think9.config import get_settings
from think9.models import Answer
from think9.retrieval.temporal import as_of_date

STRUCTURED_DECLINE = (
    "That requires the procurement tables, which this prototype does not index. "
    "The structured retriever is week 3 of the plan."
)


def build_graph(retriever, repo, llm):
    tau = get_settings().coverage_tau

    def route_node(state: BrainState) -> dict:
        route = classify(state["question"], llm)
        return {"route": route, "trace": {"route": route}}

    def retrieve_documents(state: BrainState) -> dict:
        result = retriever.retrieve(state["question"], state["route"], state["user_groups"])
        return {"retrieval": result, "trace": {"retrieval": result.trace,
                                               "coverage": result.coverage}}

    def retrieve_owner(state: BrainState) -> dict:
        owner = resolve_owner(repo, "nuvia", "procurement")
        return {"owner": owner, "trace": {"owner": owner.person_name if owner else None}}

    def synthesise_node(state: BrainState) -> dict:
        draft, citations = synthesise(llm, state["question"], state["retrieval"].chunks)
        return {"draft": draft, "citations": citations, "trace": {"draft": draft}}

    def verify_node(state: BrainState) -> dict:
        result = verify(state["draft"], state["retrieval"].chunks, llm)
        trace = {"verifier": {"stripped": result.stripped,
                              "claims": [v.__dict__ for v in result.claims]}}
        if result.refused:
            return {"answer": build_refusal(state["question"], state["retrieval"].chunks,
                                            state.get("owner")),
                    "trace": trace}
        return {
            "answer": Answer(text=result.text, outcome="answered", citations=state["citations"],
                             as_of=as_of_date(state["retrieval"].chunks)),
            "trace": trace,
        }

    def refuse_node(state: BrainState) -> dict:
        chunks = state["retrieval"].chunks if state.get("retrieval") else []
        return {"answer": build_refusal(state["question"], chunks, state.get("owner"))}

    def decline_structured(state: BrainState) -> dict:
        return {"answer": Answer(text=STRUCTURED_DECLINE, outcome="refused")}

    def after_router(state: BrainState) -> str:
        return "decline_structured" if state["route"] == "needs_structured_data" else "retrieve"

    def after_retrieval(state: BrainState) -> str:
        return "synthesise" if state["retrieval"].coverage >= tau else "refuse"

    builder = StateGraph(BrainState)
    builder.add_node("route", route_node)
    builder.add_node("retrieve_documents", retrieve_documents)
    builder.add_node("retrieve_owner", retrieve_owner)
    builder.add_node("synthesise", synthesise_node)
    builder.add_node("verify", verify_node)
    builder.add_node("refuse", refuse_node)
    builder.add_node("decline_structured", decline_structured)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route", after_router,
        {"decline_structured": "decline_structured", "retrieve": "retrieve_documents"},
    )
    # The two retrievers run in parallel; both write to `trace`, merged by the reducer.
    builder.add_edge("route", "retrieve_owner")
    builder.add_conditional_edges(
        "retrieve_documents", after_retrieval,
        {"synthesise": "synthesise", "refuse": "refuse"},
    )
    builder.add_edge("synthesise", "verify")
    builder.add_edge("verify", END)
    builder.add_edge("refuse", END)
    builder.add_edge("decline_structured", END)
    return builder.compile()


def ask(graph, question: str, user_groups: list[str], user_id: str) -> Answer:
    final = graph.invoke({
        "question": question, "user_groups": user_groups, "user_id": user_id, "trace": {},
    })
    answer = final["answer"]
    return Answer(text=answer.text, outcome=answer.outcome, citations=answer.citations,
                  as_of=answer.as_of, trace=final.get("trace", {}))
```

**Parallelism note:** `retrieve_owner` is reached by an unconditional edge from `route`, while `retrieve_documents` is reached by the conditional edge. Both fan out from `route`, so LangGraph runs them in the same superstep and the `merge_trace` reducer combines their `trace` writes. Verify this in the test by asserting `answer.trace["owner"]` is populated on the answered path; if the conditional edge suppresses the parallel branch, restructure as a single `fan_out` node returning `Send` objects to both retrievers.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_graph.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add backend/think9/agent/state.py backend/think9/agent/graph.py backend/tests/test_graph.py
git commit -m "Wire the agent as a LangGraph state machine"
```

---

## Task 17: Query log, canon write-back and HITL gates

**Files:**
- Create: `backend/think9/gates/__init__.py`, `backend/think9/gates/contested.py`, `backend/think9/gates/sensitive.py`, `backend/think9/gates/digest.py`
- Modify: `backend/think9/store/repository.py` (add `log_query`, `insert_canon`, `recent_gaps`)
- Test: `backend/tests/test_gates.py`

**Interfaces:**
- Consumes: `Repository` (Task 3), `RetrievedChunk`, `Answer` (Task 2).
- Produces: `detect_contested(chunks) -> ContestedFinding | None` with fields `attribute: str`, `values: list[tuple[str, str]]` (value, document title), `arbiter: Owner | None`; `frame_sensitive(answer, chunks) -> Answer`; `gap_digest(repo, limit=20) -> list[dict]`; repository methods `log_query(...) -> UUID`, `insert_canon(question, answer, author, source_query_id, effective_date) -> UUID`, `recent_gaps(limit) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_gates.py
from datetime import date
from uuid import uuid4

from tests.test_repository import make_document
from think9.gates.contested import detect_contested
from think9.gates.sensitive import frame_sensitive
from think9.models import Answer, RetrievedChunk

SPEC = make_document(title="Korent Spec Sheet", doc_type="spec_sheet",
                     effective_date=date(2025, 11, 2))
ANNEXE = make_document(title="Korent Contract Annexe", doc_type="contract",
                       effective_date=date(2025, 12, 9), sensitive=True)


def _chunk(doc, text):
    return RetrievedChunk(chunk_id=uuid4(), document=doc, heading_path="MOQ", text=text, score=0.8)


def test_two_live_sources_disagreeing_on_moq_are_detected():
    finding = detect_contested([
        _chunk(SPEC, "Minimum order quantity: 5,000 units."),
        _chunk(ANNEXE, "Minimum order quantity: 8,000 units."),
    ])

    assert finding is not None
    assert finding.attribute == "minimum order quantity"
    assert {v for v, _ in finding.values} == {"5,000", "8,000"}


def test_agreeing_sources_are_not_contested():
    assert detect_contested([
        _chunk(SPEC, "Minimum order quantity: 5,000 units."),
        _chunk(ANNEXE, "Minimum order quantity: 5,000 units."),
    ]) is None


def test_a_demoted_source_does_not_create_a_conflict():
    from dataclasses import replace
    demoted = replace(_chunk(ANNEXE, "Minimum order quantity: 8,000 units."), demoted=True)

    assert detect_contested([_chunk(SPEC, "Minimum order quantity: 5,000 units."), demoted]) is None


def test_a_sensitive_document_forces_evidence_framing():
    answer = Answer(text="The exclusivity term is 12 months.", outcome="answered")

    framed = frame_sensitive(answer, [_chunk(ANNEXE, "Exclusivity: 12 months.")])

    assert framed.text.startswith("Based on the sources below")
    assert framed.outcome == "answered"


def test_a_non_sensitive_answer_is_returned_unchanged():
    answer = Answer(text="Rs 22.10 per unit.", outcome="answered")
    assert frame_sensitive(answer, [_chunk(SPEC, "Rs 22.10")]) is answer
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_gates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'think9.gates.contested'`

- [ ] **Step 3: Write contested.py**

```python
# backend/think9/gates/contested.py
"""When two sources conflict and neither supersedes the other, surface both and ask."""
import re
from dataclasses import dataclass

from think9.models import Owner, RetrievedChunk

_ATTRIBUTES = {
    "minimum order quantity": re.compile(
        r"(?:minimum order quantity|MOQ)\D{0,20}([\d,]+)", re.I),
    "unit price": re.compile(r"(?:Rs|₹)\s*([\d,]+\.\d{2})", re.I),
    "lead time": re.compile(r"lead time\D{0,20}([\d,]+)\s*days", re.I),
}


@dataclass
class ContestedFinding:
    attribute: str
    values: list[tuple[str, str]]
    arbiter: Owner | None = None


def detect_contested(chunks: list[RetrievedChunk]) -> ContestedFinding | None:
    live = [c for c in chunks if not c.demoted]
    for attribute, pattern in _ATTRIBUTES.items():
        found: dict[str, str] = {}
        for chunk in live:
            match = pattern.search(chunk.text)
            if match:
                found.setdefault(match.group(1), chunk.document.title)
        if len(found) > 1:
            return ContestedFinding(attribute=attribute,
                                    values=[(v, title) for v, title in found.items()])
    return None
```

- [ ] **Step 4: Write sensitive.py and digest.py**

```python
# backend/think9/gates/sensitive.py
"""Legal, financial and people questions never return a bare answer."""
from dataclasses import replace

from think9.models import Answer, RetrievedChunk

_PREFIX = "Based on the sources below, and framed as evidence rather than a conclusion: "


def frame_sensitive(answer: Answer, chunks: list[RetrievedChunk]) -> Answer:
    if not any(c.document.sensitive for c in chunks if not c.demoted):
        return answer
    return replace(answer, text=_PREFIX + answer.text)
```

```python
# backend/think9/gates/digest.py
"""Every refusal is a line in the documentation backlog, generated from real demand."""


def gap_digest(repo, limit: int = 20) -> list[dict]:
    return repo.recent_gaps(limit)
```

Add to `Repository`:

```python
    def log_query(self, *, user_id: str, question: str, route: str, coverage_score: float,
                  outcome: str, answer_text: str, citations: list[dict],
                  as_of, trace: dict) -> UUID:
        import json
        query_id = uuid4()
        self.conn.execute(
            """INSERT INTO query_log (id, user_id, question, route, coverage_score, outcome,
                                      answer_text, citations, as_of, trace)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (query_id, user_id, question, route, coverage_score, outcome, answer_text,
             json.dumps(citations), as_of, json.dumps(trace, default=str)),
        )
        self.conn.commit()
        return query_id

    def insert_canon(self, *, question: str, answer: str, author: str,
                     source_query_id: UUID | None, effective_date) -> UUID:
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
        return [{"question": r[0], "route": r[1], "coverage": r[2], "asked_at": r[3]}
                for r in rows]
```

Add a repository test asserting `recent_gaps` returns only refused/routed rows, and that `insert_canon` round-trips.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_gates.py tests/test_repository.py -v`
Expected: all pass

- [ ] **Step 6: Wire the gates into `verify_node`**

In `graph.py`, after building the answered `Answer`: run `detect_contested` on the retrieved chunks and, if a finding exists, return `outcome="contested"` with both values and the arbiter named; then pass the answer through `frame_sensitive`. Add a graph test covering the contested path.

- [ ] **Step 7: Commit**

```bash
git add backend/think9/gates/ backend/think9/store/repository.py backend/think9/agent/graph.py backend/tests/test_gates.py
git commit -m "Add query log, canon write-back and human-in-the-loop gates"
```

---

## Task 18: FastAPI surface

**Files:**
- Create: `backend/think9/api/__init__.py`, `backend/think9/api/main.py`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: `build_graph`/`ask` (Task 16), `Repository` (Task 3), `gap_digest` (Task 17).
- Produces: `app` with `GET /health`, `POST /ask` (body `{question, user_groups, user_id}` → `{answer, outcome, citations, as_of, trace, query_id}`), `GET /trace/{query_id}`, `GET /digest`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_api.py
from fastapi.testclient import TestClient

from think9.api.main import app, get_brain
from think9.models import Answer, Citation
from datetime import date
from uuid import uuid4

CITATION = Citation(chunk_id=uuid4(), document_title="Korent Quote 2026",
                    heading_path="Pricing", deep_link="https://drive/f1",
                    effective_date=date(2026, 1, 8))


class StubBrain:
    def ask(self, question, user_groups, user_id):
        return Answer(text="Rs 22.10 per unit.", outcome="answered", citations=(CITATION,),
                      as_of=date(2026, 1, 8), trace={"route": "factual_lookup"})

    def digest(self, limit=20):
        return [{"question": "freight insurance excess", "route": "factual_lookup",
                 "coverage": 0.1, "asked_at": "2026-08-10T00:00:00Z"}]


def client() -> TestClient:
    app.dependency_overrides[get_brain] = lambda: StubBrain()
    return TestClient(app)


def test_health_reports_ok():
    assert client().get("/health").json()["status"] == "ok"


def test_ask_returns_the_answer_with_citations_and_as_of():
    response = client().post("/ask", json={"question": "amber glass price",
                                           "user_groups": ["procurement"], "user_id": "u1"})

    body = response.json()
    assert response.status_code == 200
    assert body["outcome"] == "answered"
    assert body["as_of"] == "2026-01-08"
    assert body["citations"][0]["deep_link"] == "https://drive/f1"
    assert body["trace"]["route"] == "factual_lookup"


def test_ask_rejects_an_empty_question():
    assert client().post("/ask", json={"question": "  ", "user_groups": [],
                                       "user_id": "u1"}).status_code == 422


def test_digest_lists_the_documentation_backlog():
    body = client().get("/digest").json()
    assert body["gaps"][0]["question"] == "freight insurance excess"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'think9.api.main'`

- [ ] **Step 3: Write main.py**

```python
# backend/think9/api/main.py
from functools import lru_cache

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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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
    def __init__(self) -> None:
        settings = get_settings()
        self.conn = connect(settings.database_url)
        self.repo = Repository(self.conn)
        self.llm = LLM()
        retriever = Retriever(self.conn, Embedder(), Reranker())
        self.graph = build_graph(retriever, self.repo, self.llm)

    def ask(self, question: str, user_groups: list[str], user_id: str):
        answer = graph_ask(self.graph, question, user_groups, user_id)
        self.repo.log_query(
            user_id=user_id, question=question, route=answer.trace.get("route", "unknown"),
            coverage_score=float(answer.trace.get("coverage", 0.0)), outcome=answer.outcome,
            answer_text=answer.text, citations=[c.__dict__ for c in answer.citations],
            as_of=answer.as_of, trace=answer.trace,
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


@app.post("/ask")
def ask_endpoint(request: AskRequest, brain=Depends(get_brain)) -> dict:
    answer = brain.ask(request.question, request.user_groups, request.user_id)
    return {
        "answer": answer.text,
        "outcome": answer.outcome,
        "as_of": answer.as_of.isoformat() if answer.as_of else None,
        "citations": [
            {"chunk_id": str(c.chunk_id), "document_title": c.document_title,
             "heading_path": c.heading_path, "deep_link": c.deep_link,
             "effective_date": c.effective_date.isoformat()}
            for c in answer.citations
        ],
        "trace": answer.trace,
    }


@app.get("/digest")
def digest_endpoint(brain=Depends(get_brain)) -> dict:
    return {"gaps": brain.digest()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_api.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the whole suite and lint**

Run: `cd backend && uv run ruff check . && uv run ruff format --check . && uv run pytest -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add backend/think9/api/ backend/tests/test_api.py
git commit -m "Add FastAPI surface with ask, digest and health endpoints"
```

---

## Task 19: Evaluation harness

**Files:**
- Create: `eval/metrics.py`, `eval/run.py`, `eval/questions_dev.csv`, `eval/questions_test.csv`
- Test: `backend/tests/test_metrics.py`

**Interfaces:**
- Consumes: `Brain` (Task 18), `SEEDED_FACTS` (Task 6).
- Produces: `groundedness(answer, chunks) -> float`, `refusal_precision(rows) -> float`, `refusal_recall(rows) -> float`, `recall_at_k(rows) -> float`, `as_of_correctness(rows) -> float`, `run(questions_path, out_path) -> Scorecard`.

CSV columns: `question, category, answerable, expected_substrings, must_not_contain, gold_document_title`.

**Discipline, non-negotiable:** τ, the RRF constant and the rerank cut-off are fitted on `questions_dev.csv` only. `questions_test.csv` is run **once**, after tuning is frozen. Tuning against the test set and reporting the score is fitting the test set, and with a small negative count it is easy to do by accident.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_metrics.py
from eval.metrics import as_of_correctness, groundedness, refusal_precision, refusal_recall


def _row(**kw):
    base = dict(answerable=True, outcome="answered", expected_substrings=(),
                must_not_contain=(), answer="", gold_retrieved=True, category="lookup")
    return {**base, **kw}


def test_groundedness_is_the_share_of_claims_with_a_supporting_span():
    assert groundedness(claims_supported=9, claims_total=10) == 0.9
    assert groundedness(claims_supported=0, claims_total=0) == 1.0


def test_refusal_precision_counts_only_refusals():
    rows = [
        _row(answerable=False, outcome="refused"),
        _row(answerable=True, outcome="refused"),
        _row(answerable=True, outcome="answered"),
    ]
    assert refusal_precision(rows) == 0.5


def test_refusal_recall_counts_only_unanswerable_questions():
    rows = [
        _row(answerable=False, outcome="refused"),
        _row(answerable=False, outcome="answered"),
    ]
    assert refusal_recall(rows) == 0.5


def test_as_of_correctness_fails_a_superseded_value():
    rows = [
        _row(category="temporal", answer="Rs 22.10 per unit",
             expected_substrings=("22.10",), must_not_contain=("18.40",)),
        _row(category="temporal", answer="Rs 18.40 per unit",
             expected_substrings=("22.10",), must_not_contain=("18.40",)),
    ]
    assert as_of_correctness(rows) == 0.5


def test_as_of_correctness_ignores_non_temporal_rows():
    assert as_of_correctness([_row(category="lookup")]) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.metrics'`

- [ ] **Step 3: Write metrics.py**

```python
# eval/metrics.py
def groundedness(claims_supported: int, claims_total: int) -> float:
    return 1.0 if claims_total == 0 else claims_supported / claims_total


def refusal_precision(rows: list[dict]) -> float:
    refused = [r for r in rows if r["outcome"] == "refused"]
    if not refused:
        return 1.0
    return sum(1 for r in refused if not r["answerable"]) / len(refused)


def refusal_recall(rows: list[dict]) -> float:
    unanswerable = [r for r in rows if not r["answerable"]]
    if not unanswerable:
        return 1.0
    return sum(1 for r in unanswerable if r["outcome"] == "refused") / len(unanswerable)


def recall_at_k(rows: list[dict]) -> float:
    scored = [r for r in rows if r["answerable"]]
    if not scored:
        return 1.0
    return sum(1 for r in scored if r["gold_retrieved"]) / len(scored)


def as_of_correctness(rows: list[dict]) -> float:
    temporal = [r for r in rows if r["category"] == "temporal"]
    if not temporal:
        return 1.0
    correct = sum(
        1 for r in temporal
        if all(s in r["answer"] for s in r["expected_substrings"])
        and not any(s in r["answer"] for s in r["must_not_contain"])
    )
    return correct / len(temporal)
```

- [ ] **Step 4: Author the question sets**

`eval/questions_dev.csv` — 60 rows, 40 answerable and 20 not, spread across all six categories. `eval/questions_test.csv` — 40 rows, ~28 answerable and ~12 not. Negatives are adversarial: invented vendors, plausible-but-absent figures, out-of-scope functions. Every seeded fact from Task 6 appears in the **test** set exactly once, using its `probe_question`, `expected_substrings` and `must_not_contain` verbatim.

- [ ] **Step 5: Write run.py, fit on dev, then run test once**

`run.py` loads a CSV, calls `Brain.ask` per row, computes every metric, and writes a markdown scorecard broken down by category.

```bash
cd backend && uv run python -m eval.run --questions ../eval/questions_dev.csv --out ../eval/scorecard_dev.md
# sweep COVERAGE_TAU over 0.35..0.65 in steps of 0.05, pick the best on dev, freeze it in .env
cd backend && uv run python -m eval.run --questions ../eval/questions_test.csv --out ../eval/scorecard_test.md
```

Record the frozen τ in the scorecard header. Do not re-run the test set after seeing its result.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_metrics.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add eval/ backend/tests/test_metrics.py
git commit -m "Add evaluation harness with dev-fitted threshold and held-out scorecard"
```

---

## Task 20: Ablations

**Files:**
- Create: `eval/ablations.py`
- Test: `backend/tests/test_ablations.py`

**Interfaces:**
- Consumes: `Retriever` flags (Task 12), `verify` (Task 14), `metrics` (Task 19).
- Produces: `CONFIGURATIONS: list[AblationConfig]` with fields `name`, `use_hybrid`, `use_rerank`, `use_temporal`, `use_verifier`; `run_ablations(questions_path, out_path) -> str` writing a markdown comparison table.

This table is the most persuasive artifact in the submission. It converts §2.3, §2.4 and §2.5 from assertions into measurements — including any result that comes out worse than expected.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_ablations.py
from eval.ablations import CONFIGURATIONS, render_table


def test_the_three_spec_ablations_are_all_covered():
    names = {c.name for c in CONFIGURATIONS}
    assert {"dense-only", "hybrid", "hybrid+rerank", "no-temporal", "no-verifier", "full"} <= names


def test_dense_only_disables_the_sparse_arm():
    dense_only = next(c for c in CONFIGURATIONS if c.name == "dense-only")
    assert dense_only.use_hybrid is False
    assert dense_only.use_rerank is False


def test_no_temporal_keeps_everything_else_on():
    config = next(c for c in CONFIGURATIONS if c.name == "no-temporal")
    assert config.use_temporal is False
    assert config.use_hybrid is True and config.use_verifier is True


def test_table_renders_one_row_per_configuration():
    table = render_table([
        {"name": "dense-only", "accuracy": 0.70, "groundedness": 0.91,
         "refusal_precision": 0.80, "as_of_correctness": 0.50},
        {"name": "full", "accuracy": 0.92, "groundedness": 0.98,
         "refusal_precision": 0.95, "as_of_correctness": 1.00},
    ])

    assert table.count("\n") >= 4
    assert "dense-only" in table and "full" in table
    assert "| 0.70 |" in table
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_ablations.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.ablations'`

- [ ] **Step 3: Write ablations.py**

```python
# eval/ablations.py
from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    name: str
    use_hybrid: bool
    use_rerank: bool
    use_temporal: bool
    use_verifier: bool


CONFIGURATIONS: list[AblationConfig] = [
    AblationConfig("dense-only",    False, False, True,  True),
    AblationConfig("hybrid",        True,  False, True,  True),
    AblationConfig("hybrid+rerank", True,  True,  True,  True),
    AblationConfig("no-temporal",   True,  True,  False, True),
    AblationConfig("no-verifier",   True,  True,  True,  False),
    AblationConfig("full",          True,  True,  True,  True),
]

_HEADER = (
    "| Configuration | Accuracy | Groundedness | Refusal precision | As-of correctness |\n"
    "|---|---|---|---|---|\n"
)


def render_table(rows: list[dict]) -> str:
    body = "".join(
        f"| {r['name']} | {r['accuracy']:.2f} | {r['groundedness']:.2f} | "
        f"{r['refusal_precision']:.2f} | {r['as_of_correctness']:.2f} |\n"
        for r in rows
    )
    return _HEADER + body
```

`run_ablations` loops `CONFIGURATIONS`, threads each flag into `Retriever.retrieve` and (for `use_verifier=False`) bypasses the verifier node, collects the metrics from Task 19, and writes the table.

- [ ] **Step 4: Run the ablations on the dev set**

```bash
cd backend && uv run python -m eval.ablations --questions ../eval/questions_dev.csv --out ../eval/ablations.md
```

Expected shape: `no-temporal` shows `as_of_correctness` collapse (it should quote Rs 18.40 with a valid citation); `dense-only` shows lower accuracy on entity-heavy questions. **Report whatever actually happens.** A result that contradicts the spec's prediction is a finding, not a failure — write it up in `eval/error_analysis.md`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_ablations.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add eval/ablations.py backend/tests/test_ablations.py
git commit -m "Add ablation harness for hybrid, temporal and verifier claims"
```

---

## Task 21: Web app

**Files:**
- Create: `web/` (Next.js 16, React 19, Tailwind 4), notably `web/app/page.tsx`, `web/components/AnswerCard.tsx`, `web/components/TracePanel.tsx`, `web/lib/api.ts`
- Test: `web/components/__tests__/TracePanel.test.tsx`, `web/components/__tests__/AnswerCard.test.tsx` (Vitest + Testing Library)

**Interfaces:**
- Consumes: `POST /ask` and `GET /digest` (Task 18).
- Produces: the deployed demo surface.

The trace panel is the product, not a debug view. It renders: the router's classification, the dense and sparse candidate lists side by side, the fused and reranked order, which documents the temporal layer demoted and against what, and the verifier's per-claim verdict including anything stripped.

- [ ] **Step 1: Scaffold and write the failing component tests**

```bash
npx create-next-app@latest web --typescript --tailwind --app --no-src-dir
cd web && npm i -D vitest @testing-library/react @testing-library/jest-dom jsdom
```

```tsx
// web/components/__tests__/AnswerCard.test.tsx
import { render, screen } from "@testing-library/react";
import { AnswerCard } from "../AnswerCard";

const answered = {
  answer: "Amber glass costs Rs 22.10 per unit.",
  outcome: "answered" as const,
  as_of: "2026-01-08",
  citations: [{ chunk_id: "c1", document_title: "Korent Quote 2026",
                heading_path: "Pricing", deep_link: "https://drive/f1",
                effective_date: "2026-01-08" }],
  trace: {},
};

test("shows the as-of date badge", () => {
  render(<AnswerCard result={answered} />);
  expect(screen.getByText(/as of 8 Jan 2026/i)).toBeInTheDocument();
});

test("renders each citation as a link to the original document", () => {
  render(<AnswerCard result={answered} />);
  const link = screen.getByRole("link", { name: /Korent Quote 2026/ });
  expect(link).toHaveAttribute("href", "https://drive/f1");
});

test("a refusal is visually distinct and shows no as-of badge", () => {
  render(<AnswerCard result={{ ...answered, outcome: "refused", as_of: null, citations: [] }} />);
  expect(screen.getByTestId("refusal")).toBeInTheDocument();
  expect(screen.queryByText(/as of/i)).not.toBeInTheDocument();
});
```

```tsx
// web/components/__tests__/TracePanel.test.tsx
import { render, screen } from "@testing-library/react";
import { TracePanel } from "../TracePanel";

const trace = {
  route: "factual_lookup",
  coverage: 0.82,
  retrieval: {
    dense: [{ chunk_id: "c1", rank: 1, score: 0.81, snippet: "amber glass Rs 22.10" }],
    sparse: [{ chunk_id: "c2", rank: 1, score: 0.44, snippet: "Korent SKU AMB-50-FL" }],
    fused: [{ chunk_id: "c1", rank: 1, score: 0.03, snippet: "amber glass Rs 22.10" }],
    reranked: [{ chunk_id: "c1", rank: 1, score: 5.4, snippet: "amber glass Rs 22.10" }],
    demoted: [{ title: "Korent Quote 2024", demoted_by: "doc-2026" }],
  },
  verifier: { stripped: ["Lead time is 91 days."], claims: [] },
};

test("shows the router classification", () => {
  render(<TracePanel trace={trace} />);
  expect(screen.getByText(/factual_lookup/)).toBeInTheDocument();
});

test("shows dense and sparse candidates side by side", () => {
  render(<TracePanel trace={trace} />);
  expect(screen.getByTestId("dense-column")).toHaveTextContent("amber glass Rs 22.10");
  expect(screen.getByTestId("sparse-column")).toHaveTextContent("AMB-50-FL");
});

test("names each demoted document and what superseded it", () => {
  render(<TracePanel trace={trace} />);
  expect(screen.getByText(/Korent Quote 2024/)).toBeInTheDocument();
  expect(screen.getByText(/superseded by/i)).toBeInTheDocument();
});

test("lists claims the verifier stripped", () => {
  render(<TracePanel trace={trace} />);
  expect(screen.getByText(/Lead time is 91 days/)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd web && npx vitest run`
Expected: FAIL — components do not exist

- [ ] **Step 3: Implement the components**

Build `AnswerCard` (answer text with citation markers rendered as superscript links, an as-of badge, a distinct refusal treatment carrying `data-testid="refusal"`, and a contested treatment showing both values) and `TracePanel` (collapsible; a two-column dense/sparse view, then fused, then reranked, then a demotions list reading "X — superseded by Y", then the verifier's per-claim verdicts with stripped claims struck through). `web/lib/api.ts` posts to `NEXT_PUBLIC_BACKEND_URL`.

The page header must carry, in bold, that the corpus is synthetic.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd web && npx vitest run && npx tsc --noEmit && npm run build`
Expected: 7 passed, typecheck clean, build succeeds

- [ ] **Step 5: Commit**

```bash
git add web/
git commit -m "Add web app with answer card and stage-by-stage trace panel"
```

---

## Task 22: Deploy, README, and the proposal rewrite

**Files:**
- Create: `README.md`, `render.yaml`, `.github/workflows/ci.yml`
- Modify: `Think9_Brain_Proposal.md` (§3)

**Interfaces:**
- Consumes: everything.
- Produces: the submission.

- [ ] **Step 1: Write the CI workflow**

`.github/workflows/ci.yml` runs on push and PR: `ruff check`, `ruff format --check`, `pytest` (backend), and `vitest run` + `tsc --noEmit` + `next build` (web). Add a job that runs `eval/run.py` against `questions_dev.csv` and fails if groundedness drops below the frozen dev value minus 0.02 — this is the §4.1 golden-set gate that prevents silent regressions when a prompt is "improved".

- [ ] **Step 2: Deploy the backend to Render**

`render.yaml` declares a web service (Python 3.12, `uv sync && uvicorn think9.api.main:app --host 0.0.0.0 --port $PORT`) with `DATABASE_URL`, `GOOGLE_CREDENTIALS_JSON`, `DRIVE_FOLDER_ID`, `LLM_API_KEY`, `COVERAGE_TAU` set as environment variables. Models load once at startup, not per request.

Verify: `curl https://<service>.onrender.com/health` returns `{"status":"ok"}`.

- [ ] **Step 3: Deploy the frontend to Vercel**

Set `NEXT_PUBLIC_BACKEND_URL` to the Render URL. Verify all three §3 behaviours through the deployed UI:

```
"What do we pay for 50ml amber glass?"          -> Rs 22.10, as-of 2026-01-08, 2024 quote shown demoted
"What is our freight insurance excess?"          -> refusal, nearest evidence, owner named
"Which brands buy from Korent, and on what terms?" -> both brands, composed, cited
"What is Korent's minimum order quantity?"       -> contested, both values, arbiter named
```

- [ ] **Step 4: Write the README**

Must contain, in this order: what it is; **in bold, that the corpus is synthetic**; the live URL; the three demo questions above with expected behaviour; the held-out scorecard from Task 19; the ablation table from Task 20; a "What this POC does not build, and why" section listing the structured retriever, multi-connector ingestion and the dashboard as weeks 3–4; and quickstart instructions that reproduce the eval on a clean clone.

- [ ] **Step 5: Rewrite §3 of the proposal**

Replace the borrowed Resilience numbers with this system's own scorecard and the ablation table, and link the live URL. Keep the Resilience reference as lineage — the entailment gap its error analysis identified, closed here — rather than as the evidence.

- [ ] **Step 6: Final verification**

```bash
cd backend && uv run ruff check . && uv run ruff format --check . && uv run pytest -q
cd ../web && npx vitest run && npx tsc --noEmit && npm run build
curl -s https://<service>.onrender.com/health
```

All must pass before claiming completion. If any target in §7.4 of the spec was missed, say so in the README with its error analysis — a POC that explains its own failure mode is worth more than one that hides it.

- [ ] **Step 7: Commit**

```bash
git add README.md render.yaml .github/ Think9_Brain_Proposal.md
git commit -m "Add CI, deployment config, README scorecard and proposal rewrite"
```

---

## Self-Review Notes

Checked against the spec:

- **§1 goal, three behaviours** — Tasks 14–16 build them; Task 22 Step 3 verifies all three through the deployed UI.
- **§1.1 non-goals** — `needs_structured_data` declines gracefully (Task 16); README section required in Task 22 Step 4.
- **§2 corpus and five seeded facts** — Task 6, asserted by key in Task 19 Step 4.
- **§3 data model** — Task 3, with `is_superseded` added in Task 11.
- **§4.1–4.4 retrieval** — Tasks 4, 5, 9, 10, 11, composed in 12.
- **§5 agent stages** — Tasks 13–16.
- **§6 HITL gates 2–5** — Task 17. Gate 1 is satisfied by construction: no task writes to a source system.
- **§7 evaluation** — Tasks 19 and 20, with the dev/test discipline stated as non-negotiable.
- **§8 surface** — Tasks 18 and 21.
- **§9 deployment** — Task 22.

Two things a reviewer should watch, flagged rather than hidden:

1. **Task 8's `supersedes` resolution** names its target by filename while document ids are derived from Drive file ids. The plan resolves this with a second pass and requires a test proving the two Korent quotes end up linked — this is the seam most likely to silently break the temporal demo.
2. **Task 16's parallel branch** assumes LangGraph fans out from `route` to both retrievers in one superstep. The step includes the assertion that proves it and the `Send`-based restructure to use if it does not hold.
