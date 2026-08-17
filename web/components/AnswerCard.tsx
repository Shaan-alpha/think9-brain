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

const MARKER = /^\[c:([0-9a-fA-F-]+)\]$/;

/** Swap the model's [c:<uuid>] markers for numbered references tied to the citation list. */
function renderClaim(text: string, citations: Citation[]) {
  const index = new Map(citations.map((c, i) => [c.chunk_id, i + 1]));
  const parts = text.split(/(\[c:[0-9a-fA-F-]+\])/g);
  return parts.map((part, i) => {
    const match = part.match(MARKER);
    if (!match) {
      // The model writes "...in a home setting [c:uuid]." — keeping that space left the
      // reference floating between the word it belongs to and the full stop it precedes.
      // A reference hugs its claim, so the space before one is dropped.
      const followedByMarker = parts[i + 1] !== undefined && MARKER.test(parts[i + 1]);
      return <span key={i}>{followedByMarker ? part.replace(/\s+$/, "") : part}</span>;
    }
    const n = index.get(match[1]);
    if (!n) return null;
    // Two references in a row are separated by a comma, not a margin: "1" and "2" side by
    // side read as "12", and a CSS gap does not survive being copied or read aloud.
    // Splitting on adjacent markers leaves an empty string between them, so the lookback
    // has to skip those rather than test the immediately preceding part.
    const previous = parts.slice(0, i).findLast((p) => p !== "");
    const afterMarker = previous !== undefined && MARKER.test(previous);
    return (
      <sup key={i} className="ev text-[0.65em]" style={{ color: "var(--amber)" }}>
        {afterMarker ? `,${n}` : n}
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
          <ol className="space-y-2.5">
            {result.citations.map((citation, i) => (
              <li key={citation.chunk_id} className="flex gap-2 text-sm">
                <span className="ev shrink-0" style={{ color: "var(--amber)" }}>
                  {i + 1}
                </span>
                {/* The filename and its heading path are stacked rather than sharing a
                    line. Side by side, the path was `shrink-0` and the filename was not,
                    so the filename was the only item flex could take width from: it
                    collapsed to its narrowest fitting width and broke at every hyphen,
                    turning `grove-mango-decision-memo-2025-07.md` into six stacked
                    fragments. `min-w-0` is what lets this column shrink to the card
                    instead of pushing that cost onto its contents. */}
                <div className="min-w-0">
                  <a
                    href={citation.deep_link}
                    target="_blank"
                    rel="noreferrer"
                    className="ev underline decoration-dotted underline-offset-4"
                  >
                    {citation.document_title}
                  </a>
                  {/* Shown at every width. The heading path is the span an answer is
                      grounded in and the date is what makes it current, so neither is
                      decoration that can be hidden on a phone. */}
                  <span className="ev mt-0.5 block text-xs" style={{ color: "var(--muted)" }}>
                    {citation.heading_path} · {citation.effective_date}
                  </span>
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}
    </article>
  );
}
