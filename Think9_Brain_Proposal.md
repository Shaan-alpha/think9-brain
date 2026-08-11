# The Think9 Brain

**A central institutional-memory layer for a 30-brand portfolio**

Shaan Satsangi · Think9 AI & Intelligence Challenge · August 2026

---

## 1. The Problem & Opportunity

Think9 is not 30 startups. It is one company running 30 brands, and the only structural advantage that model has over 30 independent founders is **compounding learning**: a lesson bought once by Brand 3 should be free for Brands 4 through 30.

That advantage leaks in a specific, measurable way. In a house of brands built by a shared central team, the same question gets re-answered from scratch dozens of times a month:

- *What did we last pay for 50ml amber glass, and which vendor, at what MOQ and lead time?*
- *What is our standard exclusivity clause for a creator contract, and why did we change it?*
- *Why did we kill the mango variant in Brand 7? Was it margin, sourcing, or a failed panel?*

Every one of those has an answer. It exists in a Slack thread, a meeting transcript, a signed PDF, a vendor email, or one person's head. The failure is not availability. **The failure is that retrieval costs more than re-deciding.** So people re-decide.

The compounding cost is threefold. First, direct time: senior people function as human search engines, and the answer arrives hours or days after it was needed. Second, drift: 30 brands independently arrive at 30 different answers to the same question, and the portfolio quietly turns into 30 silos wearing one logo. Third, and worst, **decision decay** — as knowledge fragments, the newest brands get the least benefit from everything learned before them, which is exactly backwards from how a portfolio should compound.

**Why this needs an agentic system, not a search bar or a wiki.**

A wiki fails because it requires someone to write the answer *before* anyone asks, and nobody does that under launch pressure. Keyword search fails because the corpus is unstructured, multi-format, and the useful questions are compositional: *"which brands have used this vendor and at what price"* is a retrieval, a join, a comparison and a synthesis, not a lookup. And a naive LLM wrapper fails worse than either, because in an operational context a **confidently wrong answer about a price, a clause, or a past decision is more expensive than no answer at all.**

What the situation actually demands is a system that retrieves across heterogeneous sources, reasons over what it finds, cites where every claim came from, knows when its evidence is insufficient, and routes to a human when it is. That is an agentic architecture with a verification stage, and it is what follows.

---

## 2. System Architecture & Workflow

### 2.1 Ingestion and provenance

Connectors pull from where knowledge already lives, rather than asking anyone to change habits: Google Drive and Docs, Slack, Gmail, meeting transcripts, contract PDFs, brand wikis, and structured exports from procurement and sales systems. Webhooks handle live changes; a scheduled backfill handles history.

Every document carries provenance from the moment it enters: source system, deep link back to the original, author, `brand_id`, function, date, document type, and the access-control list it inherited. Provenance is not metadata polish. It is what makes a citation clickable and an access rule enforceable.

### 2.2 Normalisation and chunking

Documents are converted to text with structure preserved, since headings and tables carry meaning that flat extraction destroys. Chunking is semantic rather than fixed-width, and each chunk retains a pointer to its parent section so an answer can cite the paragraph while reasoning over the whole clause.

### 2.3 Hybrid retrieval

Dense vector search alone is the wrong default for a business corpus. Vendor names, SKU codes, clause numbers and brand names are exact tokens where embeddings blur and keyword search excels. The index therefore runs **both** dense vectors and BM25 keyword search, fuses the two ranked lists with reciprocal rank fusion, and reranks the survivors with a cross-encoder.

### 2.4 The temporal authority layer

This is the part most retrieval systems omit and the part that decides whether an operations team trusts the thing after week two.

**Business facts decay.** A vendor price from 2024 is not the current price. A policy that was superseded in March is not the policy. Every fact carries an `effective_date`, documents declare what they supersede, and retrieval boosts recency within a document lineage. Answers state their as-of date explicitly. Without this, the system will confidently quote a dead price with a perfect citation, which is precisely the failure that destroys trust.

### 2.5 The agent layer

Orchestrated as a state machine (LangGraph), not a single prompt:

| Stage | Role |
|---|---|
| **Router** | Classifies the query: factual lookup, cross-brand comparison, policy question, decision archaeology ("why did we"), or one needing structured data |
| **Retrievers** (parallel) | Document retriever; structured/SQL retriever over procurement and sales tables; ownership retriever that resolves who owns a domain |
| **Synthesiser** | Composes the answer with inline citations to specific source sections |
| **Verifier** | A separate pass that checks every claim in the draft is supported by a retrieved span. Unsupported claims are stripped, or the answer is refused |
| **Refusal path** | Below a coverage threshold: *"I don't have this. The closest I have is X, and the person who would know is Y."* |

The verifier is deliberately a distinct stage rather than an instruction inside the synthesis prompt. Asking a model to check its own output in the same pass is not a control; it is a suggestion.

**Routing to the right human is itself a decision-velocity win.** A confident "I don't know, ask Priya, and here is the closest thing I found" beats twenty minutes of searching and beats a fabricated answer by an infinite margin.

### 2.6 Human-in-the-loop checkpoints

Five, each placed where the cost of being wrong is highest:

1. **Write-path gate.** The Brain never writes to a source system, sends an external message, or updates a record without explicit approval. Read is autonomous; write is always gated.
2. **Low-confidence gate.** Below threshold, the query routes to the named function owner with the draft answer attached. **Their reply is captured as a new canonical memory entry** — the system's coverage improves as a by-product of people answering questions they were going to be asked anyway.
3. **Contested-fact gate.** When two sources conflict and neither supersedes the other, the system surfaces both and asks the owner to arbitrate. The arbitration is written back as canon.
4. **Sensitive-class gate.** Legal, financial and people questions never return a bare answer. Sources are always shown, and the answer is framed as evidence rather than conclusion.
5. **Weekly gap digest.** Every refusal and low-confidence query is logged. The weekly digest of what the Brain could not answer **is the documentation backlog**, generated from real demand instead of guesswork.

### 2.7 Access control

Enforced at retrieval time, not generation time. The index is queried under the requesting user's ACL, so the model never sees a document the user could not open themselves. Filtering after generation leaks; filtering before retrieval cannot.

### 2.8 Measurement

A system like this must be judged on numbers, not demos:

- **Groundedness** on a held-out question set: share of claims traceable to a cited span
- **Refusal precision**: when it refused, should it have?
- **Time-to-answer** against the current baseline of asking a human
- **Self-serve rate**: queries resolved without pinging a person
- **Coverage**: share of function areas with indexed, current documentation

---

## 3. Proof of Concept

The prototype is built and measured. It runs the full pipeline described above — hybrid retrieval, the temporal authority layer, a LangGraph agent with a separate verifier stage — over a 64-document synthetic corpus spanning two brands and two functions, held in Postgres with pgvector.

It demonstrates the three behaviours that separate a deployable system from a demo:

1. **A cited answer.** *"What do we pay for 50ml amber glass?"* returns Rs 22.10, as of 2026-01-08, cited to the exact section of the January 2026 quote. The March 2024 quote at Rs 18.40 is retrieved, recognised as superseded, and held back.
2. **A refusal.** *"What is our standard freight insurance excess?"* is declined, with the nearest available evidence named and the question routed to the function owner. *This is the important one.*
3. **A cross-brand synthesis.** *"Which brands buy from Korent, and on what terms?"* is composed from two brands' quotes, each cited.

Two behaviours beyond the original three earned their place. Where two current sources disagree and neither supersedes the other, the system surfaces both and names the arbiter rather than choosing — and the reason that gate diverts *before* synthesis is that when the model was shown both figures it wrote *"the more recent spec sheet supersedes the earlier one"*, a supersession neither document states. The verifier caught it. That is §1's failure mode, caught by §2.5's mechanism, on a fact the corpus was built to bait.

### 3.1 What the numbers say

A 60-question development set fitted the coverage threshold; a 42-question held-out set was then run **once**, with tuning frozen.

| Metric | Held-out result |
|---|---|
| Accuracy | 0.881 |
| **Refusal recall** | **1.000** |
| Refusal precision | 0.800 |
| Recall@k | 0.867 |
| Groundedness (drafted claims surviving verification) | 0.849 |

**Across 32 unanswerable questions the system never once produced an answer.** Every error it made was over-caution or a retrieval miss; none was invention. Given that §1 rests on a confidently wrong answer costing more than no answer, that is the trade it does not make.

Three targets set before building were missed — groundedness, refusal precision, and as-of correctness — and the [error analysis](evalkit/error_analysis.md) accounts for each. The as-of miss deserves one line here because the number reads worse than the behaviour: the failing question was *refused*, not answered from stale evidence. No temporal question quoted a superseded value.

### 3.2 What the ablations prove

Each configuration toggles a real code path rather than a reimplementation.

| Configuration | recall@k | superseded doc in model's context | coverage separation |
|---|---|---|---|
| dense-only | 0.975 | 0.000 | 0.224 |
| hybrid | 0.975 | 0.000 | 0.002 |
| hybrid + rerank | **1.000** | 0.000 | **0.785** |
| temporal layer off | 1.000 | **1.000** | 0.780 |

Two results are more specific than §2 predicted. **The cross-encoder, not hybrid search, is what makes refusal possible**: the gap between mean coverage on answerable and unanswerable questions is 0.785 with reranking and 0.002 without, because reciprocal rank fusion assigns near-identical scores to everything — ordering without confidence. And **the temporal layer's contribution is exclusion rather than reordering**: the reranker already ranks the current quote first, but with the layer off the dead price sits in the model's context on every temporal question, one sentence from being quoted.

This builds on retrieval machinery I had already implemented and evaluated — a grounded RAG assistant scoring 18 of 20 on a held-out set with recall@k of 1.000. Its error analysis identified the gap this system closes: deterministic checks can verify that a figure *appears* in the retrieved text, but not that it has been *combined* correctly. The verifier's entailment stage is that missing check, and a test in this repository proves the case, using a claim whose every number is real, whose citation resolves, and which is still false.

---

## 4. Implementation Plan

### 4.1 Stack

| Layer | Choice | Why |
|---|---|---|
| Ingestion | Python workers, scheduled orchestration | Connector SDKs are Python-first |
| Store | **PostgreSQL + pgvector**, blobs in object storage | One store for metadata, vectors and ACL. A second database is a week of work and a permanent tax; add a dedicated vector DB only when scale demands it |
| Retrieval | pgvector + Postgres full-text, reciprocal rank fusion, cross-encoder rerank | Hybrid beats either alone on an entity-heavy corpus |
| Orchestration | LangGraph | Explicit state machine, inspectable transitions |
| Models | Strong LLM for synthesis, small fast model for routing, dedicated embedding model | Routing every query through the large model is the most common and most avoidable cost sink |
| Serving | FastAPI, streaming responses | |
| Surface | **Slack first**, web app second | Adoption follows habit. A tool people must remember to visit gets visited once |
| Observability | Structured logging, trace capture, error monitoring | Every answer must be reconstructable after the fact |
| Evaluation | Golden question set run in CI on every prompt or retrieval change | Prevents silent regressions when a prompt is "improved" |

### 4.2 Thirty days to a defensible MVP

**Week 1 — Beachhead and baseline.** Pick two brands and two functions (procurement and brand ops) rather than boiling the portfolio. Ingest Drive and Slack for that slice. Build a **50-question golden set from questions people actually asked last month**, and measure how long those took to answer by hand. Without this baseline, nothing later can be proven.

**Week 2 — Retrieval, citations, refusal.** Hybrid retrieval, the temporal layer, inline citations, and the refusal path. No agent yet. Measure groundedness and refusal precision against the golden set. The bar to clear before adding any complexity: **it must be right, or say it doesn't know.**

**Week 3 — The agent and the humans.** LangGraph router, verifier stage, structured retriever over procurement tables. Ship the Slack bot into one working channel. Wire the low-confidence routing so real owners start feeding the corpus.

**Week 4 — Widen and instrument.** Expand to five brands, add the contested-fact gate and the weekly gap digest, ship the metrics dashboard. Go or no-go on broader rollout decided on measured numbers, not enthusiasm.

### 4.3 The failure mode this plan is designed to avoid

The default way this project dies is indexing everything on day one, launching a company-wide oracle, and watching it produce three confidently wrong answers in week two. Trust, once lost, does not come back with a patch note.

Start narrow. Prove groundedness on a slice where the answers can be checked by the people who lived them. Earn the right to widen.

---

## 5. Why This Track

Across the five sample tracks, institutional memory is the one that **makes the other four cheaper**. A consumer-intelligence engine, a sourcing agent, a creative pipeline and a feedback hub all need the same substrate: a governed, permissioned, citable store of what this company knows, with a retrieval layer that can be trusted. Build the Brain once and the others become applications on top of it rather than four parallel integration projects.

It is also the track where the compounding is real. Every question asked makes the next answer better, every refusal writes a line in the documentation backlog, and every human arbitration becomes canon. The system gets more valuable specifically as the portfolio grows, which is the same direction Think9 is already moving.
