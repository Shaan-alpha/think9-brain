"""Ablations: turn each architectural claim into a measurement.

    uv run --project backend python -m evalkit.ablations --questions evalkit/questions_dev.csv

Everything here runs at the retrieval layer and uses no language model. That is not a
shortcut — the claims under test are retrieval claims. Whether hybrid search beats dense
alone is answered by recall@k, and whether the temporal layer works is answered by
checking which document leads the ranking. Routing those questions through a synthesiser
would add cost and variance without adding evidence.

The flags toggle real code paths in `Retriever.retrieve`. An ablation that reimplements
what it measures proves nothing.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / "backend" / ".env")

# These imports follow the .env load: settings are read at import time.
from evalkit.metrics import EvalCase
from evalkit.run import GROUPS, load_cases
from think9.agent.router import classify_deterministic
from think9.config import get_settings
from think9.retrieval.embed import Embedder
from think9.retrieval.rerank import Reranker
from think9.retrieval.retriever import Retriever
from think9.store.db import connect


@dataclass(frozen=True)
class AblationConfig:
    name: str
    use_hybrid: bool
    use_rerank: bool
    use_temporal: bool


CONFIGURATIONS: list[AblationConfig] = [
    AblationConfig("dense-only", False, False, True),
    AblationConfig("hybrid", True, False, True),
    AblationConfig("hybrid+rerank (full)", True, True, True),
    AblationConfig("no-temporal", True, True, False),
]


def _measure(retriever: Retriever, cases: list[EvalCase], config: AblationConfig) -> dict:
    gold_hits = gold_total = 0
    stale_leads = stale_in_context = temporal_total = 0
    answerable_cov: list[float] = []
    unanswerable_cov: list[float] = []

    for case in cases:
        result = retriever.retrieve(
            case.question,
            classify_deterministic(case.question),
            GROUPS,
            use_hybrid=config.use_hybrid,
            use_rerank=config.use_rerank,
            use_temporal=config.use_temporal,
        )
        (answerable_cov if case.answerable else unanswerable_cov).append(result.coverage)

        if case.answerable and case.gold_document:
            gold_total += 1
            if any(case.gold_document in c.document.title for c in result.chunks):
                gold_hits += 1

        if case.category == "temporal":
            temporal_total += 1
            # Two different failures, and only the second turns out to happen here.
            #
            # stale_lead: a superseded document at rank 1, ready to be quoted with a
            # perfectly valid citation.
            if result.chunks and result.chunks[0].document.is_superseded:
                stale_leads += 1
            # stale_in_context: a superseded document anywhere in what the synthesiser is
            # shown. The temporal layer marks these demoted and `synthesise` drops them, so
            # with the layer on the model never sees the dead price at all. With it off,
            # the dead price is in the context and one sentence away from being quoted.
            if any(c.document.is_superseded and not c.demoted for c in result.chunks):
                stale_in_context += 1

    return {
        "name": config.name,
        "recall_at_k": gold_hits / gold_total if gold_total else 1.0,
        "stale_lead_rate": stale_leads / temporal_total if temporal_total else 0.0,
        "stale_in_context_rate": stale_in_context / temporal_total if temporal_total else 0.0,
        "mean_coverage_answerable": sum(answerable_cov) / len(answerable_cov)
        if answerable_cov
        else 0.0,
        "mean_coverage_unanswerable": sum(unanswerable_cov) / len(unanswerable_cov)
        if unanswerable_cov
        else 0.0,
    }


def render(rows: list[dict]) -> str:
    header = (
        "| Configuration | recall@k | stale at rank 1 | stale in context | "
        "mean coverage (answerable) | mean coverage (unanswerable) | separation |"
    )
    lines = [header, "|---|---|---|---|---|---|---|"]
    for row in rows:
        separation = row["mean_coverage_answerable"] - row["mean_coverage_unanswerable"]
        lines.append(
            f"| {row['name']} | {row['recall_at_k']:.3f} | {row['stale_lead_rate']:.3f} | "
            f"{row['stale_in_context_rate']:.3f} | "
            f"{row['mean_coverage_answerable']:.3f} | {row['mean_coverage_unanswerable']:.3f} | "
            f"{separation:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    cases = load_cases(Path(args.questions))
    conn = connect(get_settings().database_url)
    retriever = Retriever(conn, Embedder(), Reranker())

    rows = [_measure(retriever, cases, config) for config in CONFIGURATIONS]
    conn.close()

    table = render(rows)
    print(table)
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
