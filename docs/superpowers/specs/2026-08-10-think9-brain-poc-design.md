# Think9 Brain — Proof of Concept

**Design spec** · Shaan Satsangi · 2026-08-10

Companion to [`Think9_Brain_Proposal.md`](../../../Think9_Brain_Proposal.md). The proposal
argues the architecture; this spec defines the runnable subset that proves it, and draws an
explicit line around what it does not build.

---

## 1. Goal

Demonstrate the three behaviours named in §3 of the proposal, on a deployed system a
reviewer can use, with measured numbers rather than a demo script:

1. **A cited answer** — a factual operational question answered with the exact source
   section quoted, linked, and dated.
2. **A refusal** — a question the corpus cannot support, where the system declines, names
   the closest available evidence, and identifies the owner to ask.
3. **A cross-brand synthesis** — an answer composed from two brands' documents, not
   looked up.

Two supporting artifacts carry as much weight as the behaviours themselves: a held-out
evaluation scorecard, and an ablation table that converts the proposal's architectural
claims (§2.3 hybrid retrieval, §2.4 temporal authority, §2.5 verifier) from assertions
into measurements.

### 1.1 Non-goals

Stated plainly in the README so the boundary reads as a decision, not an omission. These
are weeks 3–4 of the proposal's §4.2 plan:

- The structured/SQL retriever over procurement and sales tables. The router recognises
  queries that would need it and says so.
- Multi-connector ingestion. One connector, built properly, beats four stubs.
- The metrics dashboard. Metrics are produced by the eval harness and reported as a
  scorecard, not a live UI.
- Any write path into a source system.

---

## 2. Corpus

### 2.1 Shape

Two brands × two functions, matching the Week 1 beachhead in §4.2 of the proposal.

| Dimension | Values |
|---|---|
| Brands | `nuvia` (skincare), `grove` (home & wellness), plus `shared` for portfolio-level docs |
| Functions | `procurement`, `brand_ops` |
| Volume | 60–80 documents, roughly 700–1,200 chunks |
| Doc types | `vendor_quote`, `spec_sheet`, `contract`, `slack_thread`, `transcript`, `decision_memo`, `policy` |

The corpus is **synthetic and labelled as such in bold at the top of the README and on the
web app**. No reviewer should be able to mistake it for real Think9 data.

Documents live as real Google Docs, Sheets and PDFs in a Drive folder named
`Think9 Brain — Synthetic Corpus`, and are ingested through the Drive API. The
heterogeneity is deliberate: prose contracts with numbered clauses, tabular spec sheets,
and unstructured exported Slack threads exercise different paths through the loader.

Document bodies are LLM-generated for volume. The facts below are hand-placed, because
they are what the evaluation asserts against.

### 2.2 Seeded facts

| # | Fact | Proves | Probe question |
|---|---|---|---|
| 1 | 50ml amber glass, Vendor `Korent Glassworks`: ₹18.40/unit quoted 2024-03, superseded by ₹22.10/unit quoted 2026-01 | §2.4 temporal authority | "What do we pay for 50ml amber glass?" — the failure case quotes ₹18.40 with a valid citation |
| 2 | Two current, non-superseding sources give MOQs of 5,000 and 8,000 units for the same vendor | §2.6 contested-fact gate | "What's Korent's MOQ?" — must surface both and name the arbiter |
| 3 | Both `nuvia` and `grove` bought from `Korent` at different unit prices and payment terms | §3 cross-brand synthesis | "Which brands use Korent and on what terms?" — retrieve, join, compare |
| 4 | A plausible operational question with no supporting document anywhere in the corpus | §3 refusal | Must decline, cite nearest evidence, and route to the owner |
| 5 | A decision memo explaining why a variant was discontinued, whose supporting evidence is an older, superseded document | §4.4 route-aware temporal handling | "Why did we kill the mango variant?" — history must **not** be demoted here |

### 2.3 Owners

An `owners` table maps `(brand_id, function)` → person and contact. This is what makes a
refusal actionable rather than merely honest, and it is the input to HITL gates 2 and 3.

---

## 3. Data model

PostgreSQL (Neon) with `pgvector`. One store for vectors, full-text, metadata and ACL —
the §4.1 argument, implemented.

```sql
documents (
  id            uuid primary key,
  source_system text,          -- 'google_drive'
  source_id     text,          -- Drive file id
  deep_link     text,          -- clickable original
  title         text,
  doc_type      text,
  brand_id      text,
  function      text,
  author        text,
  created_at    timestamptz,
  effective_date date,         -- when the fact became true, not when the file was made
  supersedes_id uuid references documents(id),
  acl           text[],        -- group names inherited from the source
  sensitive     boolean,       -- drives HITL gate 4
  content_hash  text,
  ingested_at   timestamptz
)

chunks (
  id           uuid primary key,
  document_id  uuid references documents(id) on delete cascade,
  ordinal      int,
  heading_path text,           -- 'Korent Quote 2026 > Pricing > 50ml amber'
  text         text,
  embedding    vector(384),
  tsv          tsvector        -- generated from text
)

owners     (id, brand_id, function, person_name, contact, note)

query_log  (id, asked_at, user_id, question, route, coverage_score,
            outcome,           -- answered | refused | routed | contested
            answer_text, citations jsonb, as_of date, trace jsonb)

canon      (id, question, answer, author, created_at,
            source_query_id, effective_date)
```

`canon` is the write-back target for HITL gates 2 and 3: an owner's arbitration or answer
becomes a first-class retrievable document. This is the compounding loop in §2.6, made
concrete.

**Provenance is not optional metadata.** `deep_link` is what makes a citation clickable;
`acl` is what makes §2.7 enforceable; `effective_date` is what makes §2.4 possible. Those
three, plus `brand_id`, `function` and `doc_type`, are required: a document missing any of
them fails ingestion loudly rather than entering the index degraded. `supersedes_id` is
legitimately null for any document that supersedes nothing, which is most of them.

---

## 4. Retrieval

### 4.1 Chunking

Boundaries come from the document's own structure, never from a character grid — the
principle carried from `dmrag/loaders.py` in the Resilience project. Headings for Docs,
one row per record for Sheets, one message-block per turn for Slack exports. Each chunk
keeps a `heading_path` pointer to its parent section so an answer can cite the paragraph
while reasoning over the whole clause.

A document whose structure cannot be parsed degrades to a coarser unit (one chunk per
page) rather than inventing boundaries — again as in Resilience.

### 4.2 Hybrid search

| Stage | Mechanism | Output |
|---|---|---|
| Dense | `pgvector` cosine, `all-MiniLM-L6-v2` via fastembed (ONNX, no torch, no API key) | top 30 |
| Sparse | Postgres `websearch_to_tsquery` + `ts_rank_cd` | top 30 |
| Fuse | Reciprocal rank fusion, `k = 60` | merged ranking |
| Rerank | cross-encoder, ONNX, local | top 8 |

Section titles are prepended to embedded text (carried from Resilience) because several
questions nearly restate their heading; the raw body alone is what gets quoted and
grounded against, so title text cannot leak into answers.

### 4.3 Access control

Enforced at retrieval time. Every query carries the requesting user's group set, and both
the dense and sparse queries filter on `acl && :user_groups` in SQL. The model is never
shown a chunk the user could not open. Filtering after generation leaks; filtering before
retrieval cannot.

### 4.4 Temporal authority

Documents form lineages through `supersedes_id`. Default behaviour: within a lineage, only
the head is eligible, and retrieval boosts recency across lineages. Every answer states an
explicit as-of date derived from the `effective_date` of its citations.

**The layer is route-aware.** For the `decision_archaeology` route ("why did we…"),
superseded documents are exactly what the question is asking about, so demotion is
disabled and the answer instead labels each source with its effective period. A temporal
layer that unconditionally hides history would break the very query type §2.5 names.

---

## 5. Agent

LangGraph state machine. Stages are separate nodes with inspectable transitions — not one
prompt with several instructions in it.

| Stage | Behaviour |
|---|---|
| **Router** | Classifies into `factual_lookup`, `cross_brand_comparison`, `policy`, `decision_archaeology`, or `needs_structured_data`. Small fast model, with a deterministic keyword classifier as fallback. `needs_structured_data` returns a graceful "that requires the procurement tables, which this prototype does not index." |
| **Retrievers** | Run in parallel: the document retriever of §4, and an owner retriever resolving `(brand, function)` → person. |
| **Synthesiser** | Strong model, context-only instruction, mandatory inline citations to chunk ids. |
| **Verifier** | A distinct pass over the draft. Deterministic checks first — every digit-string (price, MOQ, lead time, clause number) must appear in a retrieved span, and every citation must resolve. Then a model entailment check per claim, which is what catches correctly-sourced but wrongly-combined claims. Unsupported claims are stripped; if stripping removes a load-bearing claim, the answer is refused. |
| **Refusal** | Below the coverage threshold τ on the reranked top-1: names the closest retrieved evidence and the owner to ask. |

The verifier is a separate node deliberately. Asking a model to check its own output in
the same pass is a suggestion, not a control.

**On the entailment step.** The Resilience error analysis identified precisely this gap —
its Gate 3 verifies that specifics *appear* in the context, not that they are *combined*
correctly, and its worked example ("verify shelters 72 hours before a flood [CYC-1]") is a
claim where every number is real, the citation is valid, and the answer is still wrong.
This verifier closes that gap, and the ablation in §7.3 measures whether it does.

---

## 6. Human-in-the-loop

| Gate | Status | Implementation |
|---|---|---|
| 1 — Write-path | Satisfied by construction | The system is read-only. Documented, not built. |
| 2 — Low-confidence | Built | Below τ, the query routes to the named owner with the draft attached. The owner's reply is written to `canon` and becomes retrievable. |
| 3 — Contested-fact | Built | Two live, non-superseding sources disagreeing on the same `(vendor, attribute)` → surface both with dates, name the arbiter, write the arbitration to `canon`. |
| 4 — Sensitive-class | Built (rule-based) | Triggered by `doc_type = contract` or by a `sensitive` flag set on the document at ingestion — the corpus has only two functions, so the class is carried on the document, not the function. Sources are always shown and the answer is framed as evidence rather than conclusion. |
| 5 — Gap digest | Built | A query over `query_log` for refusals and low-confidence outcomes, rendered as the documentation backlog. |

---

## 7. Evaluation

The methodology matters as much as the scores, and it mirrors the Resilience project
exactly, because that discipline is what made those numbers believable.

### 7.1 Question sets

| Set | Size | Composition | Use |
|---|---|---|---|
| Dev | ~60 | 40 answerable, 20 not | Fit τ (the §5 coverage threshold), the RRF constant and the rerank cut-off by sweep |
| Test | ~40 | ~28 answerable, ~12 not | **Run once**, after tuning is frozen |

Every question is tagged with its category — `lookup`, `temporal`, `cross_brand`,
`contested`, `archaeology`, `unanswerable` — so results break down per behaviour rather
than collapsing into one number. Negatives are written adversarially: invented vendors,
plausible-but-absent figures, out-of-scope functions.

### 7.2 Metrics

| Metric | Definition |
|---|---|
| Answer accuracy | Correct against the labelled expected answer |
| Groundedness | Share of claims traceable to a cited span |
| Refusal precision | Of refusals, the share that should have been refused |
| Refusal recall | Of unanswerable questions, the share correctly refused |
| Retrieval recall@k | Whether the gold span is in the retrieved set |
| **As-of correctness** | On temporal questions, whether the answer cites the currently-authoritative fact |

### 7.3 Ablations

| Configuration | Claim under test |
|---|---|
| dense-only → hybrid → hybrid + rerank | §2.3 — hybrid beats either alone on an entity-heavy corpus |
| temporal layer off → on | §2.4 — the off case should quote ₹18.40 with a flawless citation |
| verifier off → on | §2.5 — the entailment gap, closed and measured |

### 7.4 Targets

Not promises; the numbers reported are whatever the held-out run produces. These are the
bars that decide whether the POC is presentable:

- Groundedness ≥ 0.95
- Refusal precision ≥ 0.90
- As-of correctness = 1.00 on the `temporal` category
- Each ablation shows a measurable, explicable delta

A target missed is reported with its error analysis, in the same spirit as the Resilience
write-up. A POC that explains its own failure mode is worth more than one that hides it.

### 7.5 Tests

TDD throughout, pytest per module: RRF fusion math, temporal lineage ordering, verifier
stripping, refusal thresholds, loader boundary detection, and ACL filtering (a test that
asserts a user without a group cannot retrieve that group's chunks). The golden set runs
in CI on every prompt or retrieval change, per §4.1 of the proposal.

---

## 8. Surface

FastAPI backend with streaming responses; Next.js frontend.

The **trace panel is the product, not a debug view.** For every answer it shows:

- the router's classification and why
- dense and sparse candidate lists side by side, then the fused and reranked order
- which documents the temporal layer demoted, and against which superseding document
- the verifier's per-claim verdict, including anything stripped
- inline citations that open the exact source span, with the Drive deep link
- an explicit **as-of** date badge

This is what makes §2 watchable instead of readable.

---

## 9. Deployment

| Component | Target | Reason |
|---|---|---|
| Backend | Render | ONNX embedding and reranker weights need a warm persistent container; serverless cold starts and bundle limits make this painful on Vercel |
| Frontend | Vercel | Matches the known-good Skill-Issue setup |
| Database | Neon Postgres + pgvector | Already in use on Skill-Issue |
| Drive access | Google service account, with the corpus folder shared to it | No OAuth consent screen, no user login flow |

---

## 10. Repo layout

```
ingest/     Drive connector, normalisation, chunking
store/      schema, migrations, repository layer
retrieval/  dense, sparse, RRF, rerank, temporal
agent/      LangGraph nodes: router, retrievers, synthesiser, verifier, refusal
api/        FastAPI app
web/        Next.js frontend
eval/       question sets, metrics, ablation harness
corpus/     generators and the seeded-fact definitions
```

Each directory is independently testable and small enough to hold in context at once.

---

## 11. Plan

Seven working days.

| Day | Deliverable |
|---|---|
| 1 | Corpus authored and uploaded to Drive; service account wired; schema and migrations |
| 2 | Ingest → chunk → embed; dense and sparse retrieval with ACL filtering; unit tests |
| 3 | RRF, rerank, temporal authority layer; ablation harness skeleton |
| 4 | LangGraph agent: router, parallel retrievers, synthesiser, verifier, refusal path |
| 5 | HITL gates 2–5; `query_log`; `canon` write-back; gap digest |
| 6 | Web app with trace panel; deploy backend and frontend |
| 7 | Dev-set tuning frozen, held-out run executed once, ablation table, README, proposal §3 rewrite |

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Google service-account setup consumes half a day | Fall back to a clearly-labelled local mirror of the same folder; the connector code path is unchanged, only its source differs |
| ONNX weights make the backend slow to start | Render with a warm container; models loaded once at startup, not per request |
| Corpus authoring overruns | Bodies are generated; only the seeded facts in §2.2 are hand-written, and they are the only ones the eval asserts against |
| The week compresses | Cut in this order: contested-fact gate → cross-encoder rerank → the router's small model (falling back to the deterministic classifier). LangGraph, the temporal layer and the verifier are never cut — they are what §2 stakes its argument on |

---

## 13. Definition of done

- The three §3 behaviours are demonstrable on the deployed URL by a reviewer with no setup.
- The held-out set has been run exactly once, and its scorecard is in the README.
- The ablation table is published, including any result that came out worse than expected.
- CI is green; the golden set gates prompt and retrieval changes.
- §3 of the proposal cites this system's own numbers rather than the Resilience project's.
- The README states, in bold, that the corpus is synthetic, and lists what the POC does
  not build and why.
