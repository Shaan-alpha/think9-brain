"""Run a question set and write a scorecard.

    uv run --project backend python -m evalkit.run --questions evalkit/questions_dev.csv
    uv run --project backend python -m evalkit.run --questions evalkit/questions_dev.csv --sweep
    uv run --project backend python -m evalkit.run --questions evalkit/questions_test.csv \
        --out evalkit/scorecard_test.md

**The held-out set is run once, after tuning is frozen.** Fitting a threshold against the
set you then report is fitting the test set, and with a small number of negatives it is
easy to do by accident.
"""

import argparse
import csv
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / "backend" / ".env")

# These imports follow the .env load: settings are read at import time.
from evalkit.metrics import EvalCase, EvalRow, is_correct, scorecard
from think9.agent.graph import ask, build_graph
from think9.agent.llm import LLM
from think9.config import get_settings
from think9.retrieval.embed import Embedder
from think9.retrieval.rerank import Reranker
from think9.retrieval.retriever import Retriever
from think9.store.db import connect
from think9.store.repository import Repository

GROUPS = ["procurement", "brand_ops", "legal"]


def load_cases(path: Path) -> list[EvalCase]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            EvalCase(
                question=row["question"],
                category=row["category"],
                answerable=row["answerable"].strip().lower() == "true",
                expected_substrings=_split(row["expected_substrings"]),
                must_not_contain=_split(row["must_not_contain"]),
                gold_document=row["gold_document"].strip(),
            )
            for row in csv.DictReader(handle)
        ]


def _split(field: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in field.split(";") if part.strip())


def run_case(graph, case: EvalCase) -> EvalRow:
    answer = ask(graph, case.question, GROUPS, "eval")
    # The delivered answer is the prose plus its citations, because that is what a reader
    # receives. A vendor named only in the citation has still been communicated.
    delivered = answer.text + " " + " ".join(c.document_title for c in answer.citations)

    claims = answer.trace.get("verifier", {}).get("claims", [])
    retrieved_titles = (
        " ".join(d.get("title", "") for d in answer.trace.get("retrieval", {}).get("demoted", []))
        + " "
        + " ".join(c.document_title for c in answer.citations)
    )

    return EvalRow(
        case=case,
        outcome=answer.outcome,
        answer=answer.text,
        delivered=delivered,
        gold_retrieved=bool(case.gold_document) and case.gold_document in retrieved_titles,
        claims_supported=sum(1 for c in claims if c.get("supported")),
        claims_total=len(claims),
    )


def render(card: dict, title: str, tau: float) -> str:
    lines = [f"# {title}", "", f"Coverage threshold tau = {tau}, fitted on the dev set.", ""]
    overall = card["overall"]
    lines += [
        f"**{overall['n']} questions.**",
        "",
        "| Metric | Score |",
        "|---|---|",
    ]
    for key, value in overall.items():
        if key != "n":
            lines.append(f"| {key.replace('_', ' ')} | {value:.3f} |")
    lines += [
        "",
        "## By category",
        "",
        "| Category | n | Accuracy | Groundedness | Refusal precision |",
        "|---|---|---|---|---|",
    ]
    for category, stats in card["by_category"].items():
        lines.append(
            f"| {category} | {stats['n']} | {stats['accuracy']:.3f} | "
            f"{stats['groundedness']:.3f} | {stats['refusal_precision']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def build(tau: float | None = None):
    settings = get_settings()
    if tau is not None:
        import os

        os.environ["COVERAGE_TAU"] = str(tau)
        get_settings.cache_clear()
        settings = get_settings()
    conn = connect(settings.database_url)
    graph = build_graph(Retriever(conn, Embedder(), Reranker()), Repository(conn), LLM())
    return conn, graph, settings.coverage_tau


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out")
    parser.add_argument("--sweep", action="store_true", help="dev only: fit tau")
    parser.add_argument("--samples", action="store_true", help="print every answer")
    args = parser.parse_args()

    path = Path(args.questions)
    cases = load_cases(path)

    if args.sweep:
        if "test" in path.name:
            print("refusing to sweep on the held-out set; that is fitting the test set")
            return 2
        return sweep(cases)

    conn, graph, tau = build()
    rows = [run_case(graph, case) for case in cases]
    conn.close()

    card = scorecard(rows)
    text = render(card, f"Scorecard — {path.name}", tau)
    print(text)

    if args.samples:
        for row in rows:
            mark = "ok  " if is_correct(row) else "WRONG"
            print(f"[{mark}] ({row.case.category}) {row.case.question}")
            print(f"        -> {row.outcome}: {row.answer[:160]}")

    wrong = [r for r in rows if not is_correct(r)]
    if wrong:
        print(f"\n## {len(wrong)} incorrect\n")
        for row in wrong:
            print(f"- ({row.case.category}) {row.case.question} -> {row.outcome}")

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def sweep(cases: list[EvalCase]) -> int:
    """Fit tau on the dev set only.

    tau is the refusal threshold, and the refusal decision depends on retrieval coverage
    alone — which is deterministic and needs no model. So coverage is measured once per
    question and every candidate tau is scored against those numbers. Re-running the full
    pipeline per threshold would be several hundred model calls to learn the same thing.
    """
    from think9.agent.router import classify_deterministic
    from think9.retrieval.retriever import Retriever as _R

    settings = get_settings()
    conn = connect(settings.database_url)
    retriever = _R(conn, Embedder(), Reranker())

    measured: list[tuple[EvalCase, float]] = []
    for case in cases:
        result = retriever.retrieve(case.question, classify_deterministic(case.question), GROUPS)
        measured.append((case, result.coverage))
    conn.close()

    print("| tau | refusal precision | refusal recall | answerable wrongly refused | F1 |")
    print("|---|---|---|---|---|")
    scored: list[tuple[float, float, int]] = []
    for step in range(5, 100, 5):
        tau = step / 100
        refused = [(c, cov) for c, cov in measured if cov < tau]
        unanswerable = [c for c, _ in measured if not c.answerable]
        correct_refusals = sum(1 for c, _ in refused if not c.answerable)
        wrongly_refused = sum(1 for c, _ in refused if c.answerable)

        precision = 1.0 if not refused else correct_refusals / len(refused)
        recall = 1.0 if not unanswerable else correct_refusals / len(unanswerable)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        print(f"| {tau:.2f} | {precision:.3f} | {recall:.3f} | {wrongly_refused} | {f1:.3f} |")
        scored.append((tau, f1, wrongly_refused))

    # Differences of a few thousandths of F1 are noise on sixty questions, and the argmax
    # can sit on a threshold that refuses more answerable questions than a neighbour scoring
    # the same. Among thresholds within TOLERANCE of the best, take the one that wrongly
    # refuses fewest — refusing a question the corpus can answer is the failure this tool
    # exists to avoid.
    tolerance = 0.01
    ceiling = max(f1 for _, f1, _ in scored)
    plateau = [row for row in scored if f1_ok(row[1], ceiling, tolerance)]
    best = min(plateau, key=lambda row: (row[2], -row[1]))

    print(
        f"\nbest F1 {ceiling:.3f}; within {tolerance} of it, the threshold refusing fewest "
        f"answerable questions is tau = {best[0]:.2f} (F1 {best[1]:.3f}, "
        f"{best[2]} wrongly refused)"
    )
    print("Freeze this in backend/.env as COVERAGE_TAU before touching the held-out set.")
    return 0


def f1_ok(value: float, ceiling: float, tolerance: float) -> bool:
    return value >= ceiling - tolerance


if __name__ == "__main__":
    sys.exit(main())
