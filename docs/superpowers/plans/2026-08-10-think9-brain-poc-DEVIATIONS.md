# Deviations from the implementation plan

Companion to [`2026-08-10-think9-brain-poc.md`](2026-08-10-think9-brain-poc.md). The plan
is a record of what was decided before building; this records where building changed the
decision, and why. Tasks 1–18 are complete; 19–22 are not started.

## Defects the plan did not anticipate

Each of these was found by running the system against the real corpus and real models, not
by the test suite. That is itself a finding worth keeping: every one of them would have
shipped behind a green suite.

| # | Defect | Fix |
|---|---|---|
| 1 | **The coverage threshold was meaningless.** After reranking, `score` is a cross-encoder logit (observed −10.6 to +6.6), but τ=0.5 was tuned for cosine similarity. Correct answers and correct refusals were both accidents | Squash logits through a sigmoid so 0.5 means "more likely relevant than not" |
| 2 | **The reranker could not tell vendors apart.** Every spec sheet body reads "Minimum order quantity: N units"; only the heading names the supplier. "What is Korent's MOQ?" surfaced Sundara Caps | Rerank on `heading_path + text`, matching the rule the embedder already used |
| 3 | **A small model overrode a better classifier.** `llama-3.1-8b-instant` routed "total spend by vendor last quarter" to `factual_lookup`, sending an aggregation query into document retrieval | Deterministic patterns are high precision, so a pattern match now wins; the model only breaks ties |
| 4 | **Entailment checked claims against the whole retrieved set**, which made it strip supported claims, and would have passed a claim citing A that only B supports | Entailment reads only what the claim cites |
| 5 | **CRLF broke ingestion entirely.** The generator wrote Windows line endings; every regex anchors on `\n`, so all 64 documents failed provenance validation. The parse tests passed because they used `split("---")` | Normalise line endings at decode; generator writes LF; regression test covers a CRLF document |
| 6 | **The contested gate fired on the headline question.** ₹22.10 (Nuvia 50ml) and ₹20.75 (Grove 180ml) read as a contested price — same supplier, different product, both correct | Scope conflicts by supplier **and** brand |
| 7 | **The model fabricated a supersession.** Shown both MOQ figures it wrote "the more recent spec sheet supersedes the earlier one", which neither document states | The contested path diverts *before* synthesis, so no model gets to pick a winner |
| 8 | **The contested gate hijacked unrelated questions.** The Korent spec sheet and annexe disagree on MOQ and are retrieved for *any* Korent question, so "what neck finish does the jar use?" was answered "two sources disagree on minimum order quantity" — true, and not the question | A conflict now has to be in the attribute the question is asking about |
| 9 | **The contested gate reported the wrong supplier's conflict.** Asking Halden Glass's minimum order quantity returned Korent's disagreement: the Korent chunks are the corpus's most MOQ-shaped, so they surface for any MOQ question. Found on the held-out set — see the disclosure in the error analysis | A conflict now has to be about the supplier the question names, not only the attribute |

## Design changes

| Change | Reason |
|---|---|
| `is_superseded` added to `Document` in Task 2 rather than retrofitted in Task 11 | Avoids a mid-build schema migration the plan would have needed |
| `RetrievalResult` lives in `models.py`, not `retriever.py` | Lets the agent depend on the shape without depending on the store, which is what let Tasks 13–16 be built before the database existed |
| `infer_scope(question)` replaces the plan's hardcoded `resolve_owner(repo, "nuvia", "procurement")` | The owner retriever runs in parallel with document retrieval, so it cannot see what was retrieved. Brand and function are inferred from the question |
| `supersedes` resolves through a name→file-id index built from the folder listing | The plan proposed a second pass over document titles. Front matter names a predecessor by filename while ids derive from the source file id; an unresolvable target now raises rather than silently producing a document with no predecessor |
| `LocalFolderClient` built alongside `DriveClient` in Task 7 | The §12 fallback, so ingestion never blocked on GCP setup. Same interface, so the tested code path is identical |
| Test connection is session-scoped, truncated per test | Reconnecting per test cost 105s against Neon versus 40s |
| Ruff configured at the repo root, `docs/` excluded | `corpus/` and `eval/` sit outside `backend/`, so a config inside it was never consulted for them — and ruff was reformatting the Python blocks inside this plan document |
| Python pinned to 3.12 via `backend/.python-version` | uv resolved to 3.14 by default |
| Eval package named `evalkit`, not `eval` | A top-level package called `eval` shadows the builtin |
| The τ sweep measures retrieval coverage once per question and scores every threshold against those numbers | τ is the refusal threshold, and the refusal decision depends on coverage alone — which is deterministic and needs no model. Re-running the full pipeline per threshold was several hundred model calls to learn the same thing |
| Threshold selection takes the smallest τ within 0.01 F1 of the best, not the raw argmax | On sixty questions that difference is noise, and the argmax (0.70) refused two answerable questions where 0.55 scored the same and refused one. Refusing a question the corpus can answer is the failure this tool exists to avoid |
| Ablations measure at the retrieval layer with no model | The claims under test are retrieval claims: hybrid-vs-dense is answered by recall@k, and the temporal layer by which document leads the ranking. Routing those through a synthesiser adds cost and variance without adding evidence |

## Verifier test corrected

The plan's "wrongly combined claim" test used `8,000`, a figure absent from the corpus
entirely — so the numeric check caught it and entailment never ran. The test proved
nothing. It now uses ₹20.75, the genuine Grove rate, attached to the Nuvia jar: the
citation resolves, every digit appears in the retrieved text, all deterministic checks
pass, and only entailment catches it. A companion test asserts the same claim passes with
`llm=None`, which is what proves the first test exercises entailment rather than a cheaper
gate.

## Resolved open items

- **The corpus is 163 chunks, not the 700–1,200 the spec estimated.** The Task 20 numbers
  answered whether to enrich: hybrid's contribution over dense-only is real but small
  (recall@k 0.975 → 1.000), which is what a thin field of distractors predicts. Three of
  the five held-out failures are retrieval misses among near-duplicate documents — four
  procurement reviews and ten structurally identical spec sheets — so the useful
  enrichment is documents that differ in *substance*, not more of the same shape. Recorded
  in the error analysis rather than acted on, because acting on it after the held-out run
  would change the system the scorecard describes.
- **The plan's Task 16 note about restructuring the fan-out around `Send` is not needed.**
  Both retrievers run in one superstep, and
  `test_the_owner_retriever_runs_in_parallel_and_lands_in_the_trace` proves it.
- **`scripts/smoke.py` is in the verification loop**, not a debugging aid. It exits
  non-zero on regression and is the check that caught what the suite could not.

## Still open

- **Deployment to Render and Vercel** needs account access. `render.yaml` and the CI
  workflow are written and committed; the two account steps are not done.
- **The held-out set is spent for this version of the system.** A fix was found by reading
  its failures, so re-running it would no longer be a held-out measurement. A next
  iteration needs fresh questions.
- **Three of four §7.4 targets were missed** — groundedness, refusal precision, as-of
  correctness. Reported as they came out; see `evalkit/error_analysis.md` for the per-failure
  account and what would actually move them.
