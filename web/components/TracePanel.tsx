"use client";

import { useState } from "react";
import type { Candidate, Trace } from "@/lib/types";

function Column({
  title,
  note,
  candidates,
  testid,
}: {
  title: string;
  note: string;
  candidates: Candidate[];
  testid?: string;
}) {
  return (
    <div data-testid={testid}>
      <p className="label">{title}</p>
      <p className="mb-2 text-xs" style={{ color: "var(--muted)" }}>
        {note}
      </p>
      {candidates.length === 0 ? (
        <p className="ev text-xs" style={{ color: "var(--stale)" }}>
          not run
        </p>
      ) : (
        <ol className="space-y-1">
          {candidates.slice(0, 5).map((c) => (
            <li key={`${c.chunk_id}-${c.rank}`} className="flex gap-2 text-xs">
              <span className="ev shrink-0" style={{ color: "var(--muted)" }}>
                {c.rank}
              </span>
              <span className="ev shrink-0" style={{ color: "var(--amber)" }}>
                {c.score.toFixed(3)}
              </span>
              <span className="truncate">{c.snippet}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

export function TracePanel({ trace }: { trace: Trace }) {
  const [open, setOpen] = useState(false);
  const retrieval = trace.retrieval ?? {};
  const verifier = trace.verifier ?? {};
  const stripped = new Set(verifier.stripped ?? []);

  return (
    <section className="mt-4">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="label flex items-center gap-2 py-2"
        aria-expanded={open}
      >
        <span aria-hidden>{open ? "−" : "+"}</span>
        {open ? "Hide trace" : "Show trace"}
      </button>

      {open && (
        <div
          className="stage rounded-sm border p-5 sm:p-6"
          style={{ background: "var(--card)", borderColor: "var(--rule)" }}
        >
          <div className="mb-6 flex flex-wrap gap-x-8 gap-y-2">
            <div>
              <p className="label">Route</p>
              <p data-testid="route" className="ev text-sm">
                {trace.route ?? "—"}
              </p>
            </div>
            <div>
              <p className="label">Coverage</p>
              <p data-testid="coverage" className="ev text-sm">
                {trace.coverage === undefined ? "—" : trace.coverage.toFixed(3)}
              </p>
            </div>
            <div>
              <p className="label">Owner on call</p>
              <p className="ev text-sm">{trace.owner ?? "none"}</p>
            </div>
          </div>

          <div className="mb-6 grid gap-6 border-t pt-4 sm:grid-cols-2" style={{ borderColor: "var(--rule)" }}>
            <Column
              title="Dense"
              note="Meaning. Blurs on exact tokens."
              candidates={retrieval.dense ?? []}
              testid="dense-column"
            />
            <Column
              title="Sparse"
              note="Exact tokens. Vendor names, SKUs, clause numbers."
              candidates={retrieval.sparse ?? []}
              testid="sparse-column"
            />
          </div>

          <div className="mb-6 grid gap-6 border-t pt-4 sm:grid-cols-2" style={{ borderColor: "var(--rule)" }}>
            <Column
              title="Fused"
              note="Reciprocal rank fusion. Ordering only — the scores carry no confidence."
              candidates={retrieval.fused ?? []}
            />
            <Column
              title="Reranked"
              note="Cross-encoder. This is where the confidence signal comes from."
              candidates={retrieval.reranked ?? []}
            />
          </div>

          <div data-testid="demotions" className="mb-6 border-t pt-4" style={{ borderColor: "var(--rule)" }}>
            <p className="label">Temporal authority</p>
            {(retrieval.demoted ?? []).length === 0 ? (
              <p className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
                Nothing retrieved was superseded.
              </p>
            ) : (
              <ul className="mt-2 space-y-1">
                {(retrieval.demoted ?? []).map((d) => (
                  <li key={d.title} className="text-xs">
                    <span className="ev line-through" style={{ color: "var(--stale)" }}>
                      {d.title}
                    </span>
                    <span style={{ color: "var(--muted)" }}>
                      {" "}
                      — superseded, effective <span className="ev">{d.effective_date}</span>. Held
                      back from the answer.
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div data-testid="verifier" className="border-t pt-4" style={{ borderColor: "var(--rule)" }}>
            <p className="label">Verifier</p>
            <p className="mb-2 text-xs" style={{ color: "var(--muted)" }}>
              A separate pass. Each claim is checked against the span it cites.
            </p>
            {(verifier.claims ?? []).length === 0 ? (
              <p className="text-xs" style={{ color: "var(--muted)" }}>
                No claims to check.
              </p>
            ) : (
              <ul className="space-y-2">
                {(verifier.claims ?? []).map((claim, i) => {
                  const wasStripped = !claim.supported || stripped.has(claim.claim);
                  return (
                    <li key={i} className="text-xs">
                      <span
                        className={wasStripped ? "line-through" : undefined}
                        style={{ color: wasStripped ? "var(--stale)" : "var(--ink)" }}
                      >
                        {claim.claim}
                      </span>
                      <span
                        className="ev ml-2"
                        style={{ color: wasStripped ? "var(--contest)" : "var(--muted)" }}
                      >
                        {claim.reason}
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
