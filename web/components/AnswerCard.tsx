"use client";

import type { AskResponse, Citation } from "@/lib/types";

const OUTCOME_COPY: Record<string, { label: string; note: string }> = {
  answered: { label: "Answered", note: "Every claim below is traceable to a cited span." },
  refused: {
    label: "Refused",
    note: "The corpus does not support an answer. The nearest evidence and the owner are named.",
  },
  contested: {
    label: "Contested",
    note: "Two current sources disagree and neither supersedes the other.",
  },
  routed: { label: "Routed", note: "Sent to the function owner to answer." },
};

function formatDate(iso: string): string {
  return new Date(iso + "T00:00:00Z").toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}

/** Swap the model's [c:<uuid>] markers for numbered references tied to the citation list. */
function renderClaim(text: string, citations: Citation[]) {
  const index = new Map(citations.map((c, i) => [c.chunk_id, i + 1]));
  const parts = text.split(/(\[c:[0-9a-fA-F-]+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/^\[c:([0-9a-fA-F-]+)\]$/);
    if (!match) return <span key={i}>{part}</span>;
    const n = index.get(match[1]);
    if (!n) return null;
    return (
      <sup key={i} className="ev ml-0.5 text-[0.65em]" style={{ color: "var(--amber)" }}>
        {n}
      </sup>
    );
  });
}

export function AnswerCard({ result }: { result: AskResponse }) {
  const copy = OUTCOME_COPY[result.outcome] ?? OUTCOME_COPY.answered;
  const accent = result.outcome === "contested" ? "var(--contest)" : "var(--slate)";
  const wash = result.outcome === "answered" ? "var(--amber-wash)" : "var(--slate-wash)";

  return (
    <article
      className="rounded-sm border p-6 sm:p-8"
      style={{ background: "var(--card)", borderColor: "var(--rule)" }}
    >
      <header className="mb-5 flex flex-wrap items-center gap-3">
        <span
          data-testid="outcome"
          className="label rounded-sm px-2 py-1"
          style={{ background: wash, color: accent }}
        >
          {copy.label}
        </span>
        {result.as_of && (
          <span data-testid="as-of" className="ev text-xs" style={{ color: "var(--amber)" }}>
            as of {formatDate(result.as_of)}
          </span>
        )}
      </header>

      <p
        data-testid="answer-text"
        className="text-[1.0625rem] leading-relaxed sm:text-lg"
        style={{ maxWidth: "62ch" }}
      >
        {renderClaim(result.answer, result.citations)}
      </p>

      <p className="mt-3 text-sm" style={{ color: "var(--muted)", maxWidth: "62ch" }}>
        {copy.note}
      </p>

      {result.trace.contested && (
        <div
          data-testid="contested"
          className="mt-6 border-t pt-4"
          style={{ borderColor: "var(--rule)" }}
        >
          <p className="label mb-3">In conflict — {result.trace.contested.attribute}</p>
          <div className="grid gap-3 sm:grid-cols-2">
            {result.trace.contested.values.map(([value, source]) => (
              <div
                key={source}
                className="rounded-sm border p-3"
                style={{ borderColor: "var(--rule)" }}
              >
                <p className="ev text-xl" style={{ color: "var(--contest)" }}>
                  {value}
                </p>
                <p className="ev mt-1 text-xs" style={{ color: "var(--muted)" }}>
                  {source}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {result.citations.length > 0 && (
        <div className="mt-6 border-t pt-4" style={{ borderColor: "var(--rule)" }}>
          <p className="label mb-2">Sources</p>
          <ol className="space-y-1.5">
            {result.citations.map((citation, i) => (
              <li key={citation.chunk_id} className="flex gap-2 text-sm">
                <span className="ev shrink-0" style={{ color: "var(--amber)" }}>
                  {i + 1}
                </span>
                <a
                  href={citation.deep_link}
                  target="_blank"
                  rel="noreferrer"
                  className="ev underline decoration-dotted underline-offset-4"
                >
                  {citation.document_title}
                </a>
                <span className="ev hidden shrink-0 sm:inline" style={{ color: "var(--muted)" }}>
                  › {citation.heading_path} · {citation.effective_date}
                </span>
              </li>
            ))}
          </ol>
        </div>
      )}
    </article>
  );
}
