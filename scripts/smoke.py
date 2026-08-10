"""End-to-end check of the behaviours the prototype claims.

    uv run --project backend python scripts/smoke.py

Hits the real database and the real models — this is the check that catches what unit
tests cannot, and every defect found late in this build was found here rather than in the
suite. Exits non-zero if any behaviour regresses.

The probe questions come from corpus.seeds so they cannot drift from the corpus they are
asserting against.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
# Running as a script puts scripts/ on the path, not the repo root, so `corpus` would not
# resolve. pytest gets this from the pythonpath setting in backend/pyproject.toml.
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / "backend" / ".env")

# These imports must follow the two lines above: they read settings at import time, and
# `corpus` is not importable until the path is set.
from corpus.seeds import fact
from think9.agent.graph import ask, build_graph
from think9.agent.llm import LLM
from think9.config import get_settings
from think9.retrieval.embed import Embedder
from think9.retrieval.rerank import Reranker
from think9.retrieval.retriever import Retriever
from think9.store.db import connect
from think9.store.repository import Repository

# Every group, because the point here is behaviour rather than access control; the ACL
# path has its own tests.
GROUPS = ["procurement", "brand_ops", "legal"]

# (label, question, expected outcome, substrings the answer must contain)
CASES: list[tuple[str, str, str, tuple[str, ...]]] = [
    (
        "cited answer (temporal)",
        fact("amber_glass_price").probe_question,
        "answered",
        fact("amber_glass_price").expected_substrings,
    ),
    ("refusal", fact("unanswerable_gap").probe_question, "refused", ()),
    (
        "cross-brand synthesis",
        fact("korent_cross_brand").probe_question,
        "answered",
        fact("korent_cross_brand").expected_substrings,
    ),
    (
        "contested fact",
        fact("korent_moq_contested").probe_question,
        "contested",
        fact("korent_moq_contested").expected_substrings,
    ),
    (
        "decision archaeology",
        fact("mango_variant_archaeology").probe_question,
        "answered",
        (),
    ),
    (
        "structured decline",
        "Show me total spend by vendor last quarter",
        "refused",
        ("procurement tables",),
    ),
]

FORBIDDEN = {"cited answer (temporal)": fact("amber_glass_price").must_not_contain}


def main() -> int:
    settings = get_settings()
    conn = connect(settings.database_url)
    graph = build_graph(Retriever(conn, Embedder(), Reranker()), Repository(conn), LLM())

    failures: list[str] = []
    for label, question, expected_outcome, expected_substrings in CASES:
        answer = ask(graph, question, GROUPS, "smoke")
        # What the reader sees is the prose and its citations together, so both count as
        # the delivered answer. "What do we pay for amber glass?" never asks who the
        # supplier is, and requiring the vendor name in the prose would fail an answer
        # that names it in the citation — which is where it belongs.
        delivered = answer.text + " " + " ".join(c.document_title for c in answer.citations)

        problems = []
        if answer.outcome != expected_outcome:
            problems.append(f"outcome {answer.outcome!r}, expected {expected_outcome!r}")
        for needle in expected_substrings:
            if needle.lower() not in delivered.lower():
                problems.append(f"missing {needle!r}")
        for needle in FORBIDDEN.get(label, ()):
            if needle in answer.text:
                problems.append(f"quoted superseded value {needle!r}")

        status = "FAIL" if problems else "ok"
        print(f"[{status:4s}] {label}")
        print(f"        Q: {question}")
        print(
            f"        route={answer.trace.get('route')} outcome={answer.outcome} "
            f"as_of={answer.as_of}"
        )
        print(f"        A: {answer.text[:220]}")
        for citation in answer.citations[:3]:
            print(f"           - {citation.document_title} > {citation.heading_path}")
        demoted = answer.trace.get("retrieval", {}).get("demoted", [])
        if demoted:
            print(f"        demoted: {[d['title'] for d in demoted]}")
        stripped = answer.trace.get("verifier", {}).get("stripped", [])
        if stripped:
            print(f"        verifier stripped {len(stripped)} claim(s)")
        for problem in problems:
            print(f"        !! {problem}")
            failures.append(f"{label}: {problem}")
        print()

    conn.close()
    if failures:
        print(f"{len(failures)} failure(s)")
        return 1
    print(f"all {len(CASES)} behaviours ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
