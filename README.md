# The Think9 Brain — proof of concept

An institutional-memory assistant for a house of brands. It answers operational questions
from the documents a company already has, cites the exact section it used, states the date
that answer is good as of, and declines when the corpus cannot support an answer.

Companion to [`Think9_Brain_Proposal.md`](Think9_Brain_Proposal.md), which argues the
architecture. This is the runnable subset that tests it.

### ▶ Try it: **<https://think9-brain.vercel.app>**

Six one-click questions cover every behaviour below. Each answer opens its working — what
each retrieval arm found, what the temporal layer held back, and what the verifier struck
out with the reason.

**API:** <https://think9-brain-api.onrender.com> — `/health` for status, `POST /ask` to
query, `GET /digest` for the questions it could not answer.

```bash
curl -s https://think9-brain-api.onrender.com/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What do we pay for 50ml amber glass?","user_groups":["procurement"]}'
```

Hosted on Render's free tier, which sleeps after 15 minutes idle. The first request after
a sleep waits for the container and the ONNX weights to load — around a minute. Every
request after that is warm.

> **The corpus is synthetic.** Brands, vendors, people, prices and documents are invented
> for this prototype. No real Think9 data appears anywhere in it. The generator is
> [`corpus/generate.py`](corpus/generate.py); see [`corpus/README.md`](corpus/README.md).

---

## What it does, and what that looked like

Six behaviours, each runnable in one click from the web app. These are the actual outputs
from [`scripts/smoke.py`](scripts/smoke.py), which asserts all six and exits non-zero on
regression.

| Ask | Outcome |
|---|---|
| *What do we pay for 50ml amber glass?* | **Rs 22.10**, as of 2026-01-08, cited to `korent-quote-2026-01 › Pricing`. The 2024 quote at Rs 18.40 was retrieved and held back |
| *What is our standard freight insurance excess for sea shipments?* | **Refused.** Names the nearest evidence and routes to Arun Menon |
| *Which brands buy from Korent, and on what terms?* | **Composed** from two brands' quotes, both cited |
| *What is Korent's minimum order quantity?* | **Contested.** Surfaces both 5,000 and 8,000 with their sources and names the arbiter |
| *Why did we discontinue the mango variant?* | **Answered from a superseded document** — history is what the question is about |
| *Show me total spend by vendor last quarter* | **Declined.** "That requires the procurement tables, which this prototype does not index" |

The fourth row is the one worth dwelling on. Shown both conflicting figures, the model
wrote *"the more recent spec sheet supersedes the earlier one"* — a supersession neither
document states. The verifier stripped it. That is the §1 failure mode, caught by the §2.5
mechanism, on a fact the corpus was built to bait.

---

## Results

The held-out set was run **once**, at τ = 0.55 fitted on the dev set with tuning frozen
beforehand. **Three of the four targets in the spec were missed.** They are reported as
they came out; the full account is in [`evalkit/error_analysis.md`](evalkit/error_analysis.md).

| Metric | Target | Dev (60 q) | Held-out (42 q) | |
|---|---|---|---|---|
| Accuracy | — | 0.900 | 0.881 | |
| Groundedness | ≥ 0.95 | 0.848 | 0.849 | **missed** |
| Refusal precision | ≥ 0.90 | 0.870 | 0.800 | **missed** |
| Refusal recall | — | 1.000 | **1.000** | |
| Recall@k | — | 0.925 | 0.867 | |
| As-of correctness | 1.00 | 1.000 | 0.667 | **missed** |

**Refusal recall is 1.000 on both sets.** Across 32 unanswerable questions — invented
vendors, plausible-but-absent figures, out-of-scope functions — the system never once
produced an answer. Every miss is over-caution or a retrieval failure; none is invention.

Two numbers need reading carefully, and both are argued in the error analysis:

- **Groundedness is measured over drafted claims, not delivered ones.** Unsupported claims
  are stripped before an answer returns, so delivered groundedness is 1.000 by
  construction. 0.849 is the share of what the model *proposed* that survived — a measure
  of how much work the verifier is doing.
- **The as-of miss is a refusal, not a stale answer.** Zero temporal questions quoted a
  superseded value. The metric scores a refusal as incorrect, which is right for decision
  velocity, but the failure §2.4 exists to prevent did not occur.

### Ablations

Each flag toggles a real code path in `Retriever.retrieve`. An ablation that reimplements
what it measures proves nothing.

| Configuration | recall@k | stale in context | coverage separation |
|---|---|---|---|
| dense-only | 0.975 | 0.000 | 0.224 |
| hybrid | 0.975 | 0.000 | 0.002 |
| hybrid + rerank (full) | **1.000** | 0.000 | **0.785** |
| no-temporal | 1.000 | **1.000** | 0.780 |

Two findings sharper than the proposal predicted:

**The cross-encoder, not hybrid search, is what makes refusal possible.** Coverage
separation — the gap between mean coverage on answerable and unanswerable questions — is
0.785 with reranking and 0.002 without. Reciprocal rank fusion assigns nearly identical
scores to everything, so fused scores carry ordering but no confidence. Without the
reranker no threshold could separate "I know this" from "I don't".

**The temporal layer's job is exclusion, not reordering.** Whether a superseded document
*leads* the ranking is 0.000 in every configuration — the reranker orders correctly
unaided, so the obvious metric shows nothing. But demoted chunks are dropped from the
synthesiser's context, and turning the layer off puts a superseded document in front of
the model on **100%** of temporal questions, one sentence from being quoted.

Hybrid's own contribution is real but modest here (recall@k 0.975 → 1.000). At 163 chunks
there are few near-miss distractors for the sparse arm to rescue.

---

## How it works

```
Drive folder ──▶ connector ──▶ structure-derived chunking ──▶ Postgres + pgvector
                                  (provenance, effective_date, supersedes, acl)

question ──▶ ROUTER ──┬──▶ dense (pgvector) ─┐
                      │    sparse (Postgres FTS) ─┴─▶ RRF ─▶ cross-encoder ─▶ TEMPORAL
                      └──▶ owner retriever                                       │
                                                    coverage < τ ──▶ REFUSE ◀────┤
                                                    conflict ──▶ CONTESTED ◀─────┤
                                                                SYNTHESISE ◀─────┘
                                                                     │
                                                                 VERIFIER
                                                    (citations, numbers, entailment)
```

Decisions worth reading the code for:

- **Access control is enforced in SQL, at retrieval time** ([`search.py`](backend/think9/retrieval/search.py)).
  The model is never shown a chunk the asking user could not open. Filtering after
  generation leaks; filtering before retrieval cannot.
- **The verifier is a separate stage** ([`verifier.py`](backend/think9/agent/verifier.py)),
  not an instruction inside the synthesis prompt. It runs cheapest-first: citation
  validity, then numeric grounding, then entailment against the span the claim cites.
  Entailment is what catches a claim built from real figures combined into a relationship
  the evidence never asserts.
- **The temporal layer is route-aware** ([`temporal.py`](backend/think9/retrieval/temporal.py)).
  For "why did we…" questions, superseded documents are exactly what is being asked about,
  so demotion is disabled. A layer that unconditionally hid history would break the one
  query type that needs it.
- **Contested facts divert before synthesis** ([`contested.py`](backend/think9/gates/contested.py)).
  Given two conflicting figures a model reliably invents a reason to prefer one.

**Stack.** Python 3.12 · Postgres + pgvector (Neon) · fastembed ONNX, no torch ·
LangGraph · FastAPI · Next.js 16 + Tailwind 4 · Groq (`llama-3.3-70b-versatile` for
synthesis and verification, `llama-3.1-8b-instant` for routing).

---

## What this does not build, and why

Weeks 3–4 of the proposal's plan, left out deliberately rather than gestured at:

- **The structured/SQL retriever** over procurement and sales tables. The router
  recognises queries that would need it and says so rather than guessing.
- **Multi-connector ingestion.** One connector built properly beats four stubs. The Drive
  client and the local-mirror fallback share an interface, so the tested code path is
  identical whichever is configured.
- **The metrics dashboard.** Metrics are produced by the eval harness and reported here.

---

## Running it

Requires Postgres with `pgvector` (a free Neon project is enough) and, for generated
answers, an OpenAI-compatible API key. **The retrieval evaluation and every ablation run
without any API key** — embeddings and reranking are local ONNX.

```bash
cp backend/.env.example backend/.env      # fill in DATABASE_URL, TEST_DATABASE_URL, LLM_API_KEY

python corpus/generate.py                 # writes the 64-document synthetic corpus
uv run --project backend python -m think9.ingest    # ingest, embed, link supersessions

uv run --project backend python scripts/smoke.py    # all six behaviours, exits non-zero on regression
```

Backend and frontend:

```bash
cd backend && uv run uvicorn think9.api.main:app --reload --port 8000
cd web && npm install && npm run dev        # reads NEXT_PUBLIC_BACKEND_URL
```

Reproducing the evaluation:

```bash
uv run --project backend python -m evalkit.run --questions evalkit/questions_dev.csv --sweep
uv run --project backend python -m evalkit.run --questions evalkit/questions_dev.csv
uv run --project backend python -m evalkit.ablations --questions evalkit/questions_dev.csv
```

The held-out set is deliberately not part of that loop. Fitting a threshold against the set
you then report is fitting the test set, and with a small number of negatives it is easy to
do by accident.

Tests and lint:

```bash
cd backend && uv run ruff check .. && uv run ruff format --check .. && uv run pytest -q
cd web && npm run lint && npm run typecheck && npm test && npm run build
```

Postgres-dependent tests skip when `TEST_DATABASE_URL` is unset, so the suite runs on a
clean clone. `TEST_DATABASE_URL` must point at a different branch from `DATABASE_URL` — the
fixture truncates.

---

## Documentation

| File | What it holds |
|---|---|
| [`Think9_Brain_Proposal.md`](Think9_Brain_Proposal.md) | The argument: problem, architecture, plan |
| [`docs/superpowers/specs/`](docs/superpowers/specs/) | The design spec this was built against |
| [`docs/superpowers/plans/`](docs/superpowers/plans/) | The task-by-task plan, and a record of where building changed it |
| [`evalkit/error_analysis.md`](evalkit/error_analysis.md) | Held-out results, per-failure |
| [`corpus/README.md`](corpus/README.md) | The synthetic corpus and its five seeded facts |

A note on the last one. Every defect found after the retrieval stack was complete was found
by running the system against the real corpus, not by the test suite — which was green
throughout. Threshold calibration, score-scale mismatches, prompt-level model behaviour and
real-data encoding all live outside what unit tests assert. That is why `scripts/smoke.py`
exists and why it is part of the verification loop rather than a debugging aid.
