# Synthetic corpus

**This corpus is entirely synthetic. No real Think9 data appears anywhere in it.** Brands,
vendors, people, prices and documents are invented for the purpose of demonstrating and
measuring the Think9 Brain prototype.

Regenerate with:

```bash
uv run --project backend python corpus/generate.py    # writes corpus/out/
```

`corpus/out/` is gitignored — the generator is the source of truth, not its output.

## Shape

64 documents across two brands (`nuvia`, `grove`) plus portfolio-level `shared`, two
functions (`procurement`, `brand_ops`), and all seven document types: vendor quotes, spec
sheets, contracts, exported Slack threads, meeting transcripts, decision memos and
policies.

## The five seeded facts

Document bodies are templated. The facts below are hand-placed in [`generate.py`](generate.py),
because they are what the evaluation asserts against. Each is guarded by a test in
`backend/tests/test_corpus_generate.py`.

| Fact | Documents | What it proves |
|---|---|---|
| Amber glass price | `korent-quote-2024-03.md` (Rs 18.40) superseded by `korent-quote-2026-01.md` (Rs 22.10) | Temporal authority. Without it the system quotes the dead price with a perfect citation |
| Korent MOQ | `korent-spec-sheet-2025-11.md` (5,000) vs `korent-contract-annexe-2025-12.md` (8,000), neither superseding the other | The contested-fact gate |
| Korent cross-brand | `korent-quote-2026-01.md` (Nuvia) and `korent-quote-grove-2025-09.md` (Grove) | Cross-brand synthesis — retrieve, join, compare |
| Freight insurance excess | **no document** | The refusal path. `test_the_seeded_gap_is_really_a_gap` fails the build if any document ever mentions it |
| Mango variant | `grove-mango-decision-memo-2025-07.md` resting on the superseded `grove-consumer-panel-2025-05.md` | Route-aware temporal handling — history must not be demoted for a "why did we" question |

The fourth row is the one that needs guarding rather than authoring. A refusal demo is
worthless if the corpus quietly answers the question, so the test greps every generated
file for the probe terms and fails if any of them appears.
