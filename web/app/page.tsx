"use client";

import { useEffect, useRef, useState } from "react";
import { AnswerCard } from "@/components/AnswerCard";
import { TracePanel } from "@/components/TracePanel";
import { AskAborted, ask } from "@/lib/api";
import type { AskResponse } from "@/lib/types";

/* The behaviours the prototype claims, as one click each. */
const PROBES: { question: string; shows: string }[] = [
  { question: "What do we pay for 50ml amber glass?", shows: "cited answer, dated" },
  {
    question: "What is our standard freight insurance excess for sea shipments?",
    shows: "refusal, routed to an owner",
  },
  {
    question: "Which brands buy from Korent, and on what terms?",
    shows: "composed across two brands",
  },
  { question: "What is Korent's minimum order quantity?", shows: "sources in conflict" },
  { question: "Why did we discontinue the mango variant?", shows: "decision archaeology" },
  { question: "Show me total spend by vendor last quarter", shows: "declined, out of scope" },
];

export default function Home() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  /* The request currently allowed to write to the page.
   *
   * Asking a second question while the first is still running used to leave both in
   * flight against a single small instance: they slowed each other down, and whichever
   * finished last won the render — so the answer on screen could belong to the question
   * above it. The newest question wins, and the one it replaced is cancelled rather than
   * left to compete for the same instance. */
  const inFlight = useRef<AbortController | null>(null);

  useEffect(() => () => inFlight.current?.abort(), []);

  async function run(q: string) {
    if (!q.trim()) return;

    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    const isCurrent = () => inFlight.current === controller;

    setQuestion(q);
    setPending(true);
    setError(null);
    setNotice(null);
    setResult(null);

    try {
      const answer = await ask(q, {
        signal: controller.signal,
        onRetry: (attempt, reason) => {
          if (!isCurrent()) return;
          setNotice(
            `Attempt ${attempt} — ${reason}. Waking the API and trying again; this is the ` +
              "free plan starting from cold, not a failure yet.",
          );
        },
      });
      if (!isCurrent()) return;
      setResult(answer);
    } catch (e) {
      // A cancelled request has already been replaced by a newer one. Reporting it would
      // show an error for something the visitor deliberately moved on from.
      if (e instanceof AskAborted || !isCurrent()) return;
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      if (isCurrent()) {
        setPending(false);
        setNotice(null);
      }
    }
  }

  return (
    <main className="mx-auto max-w-3xl px-5 py-12 sm:px-8 sm:py-16">
      <header className="mb-10">
        <p className="label">Think9 · institutional memory</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">The Brain</h1>
        <p className="mt-3 text-base" style={{ color: "var(--muted)", maxWidth: "58ch" }}>
          Answers operational questions from the documents the company already has, cites the
          exact section it used, and says so when the corpus cannot support an answer.
        </p>
        <p
          className="mt-4 rounded-sm border px-3 py-2 text-sm"
          style={{ borderColor: "var(--rule)", color: "var(--muted)" }}
        >
          <strong style={{ color: "var(--ink)" }}>The corpus is synthetic.</strong> Brands,
          vendors, people and figures are invented for this prototype. No real Think9 data
          appears anywhere in it.
        </p>
      </header>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void run(question);
        }}
        className="mb-4"
      >
        <label htmlFor="q" className="label">
          Ask
        </label>
        <div className="mt-2 flex gap-2">
          <input
            id="q"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What do we pay for 50ml amber glass?"
            className="ev min-w-0 flex-1 rounded-sm border px-3 py-2.5 text-sm"
            style={{
              background: "var(--card)",
              borderColor: "var(--rule)",
              color: "var(--ink)",
            }}
          />
          <button
            type="submit"
            disabled={pending}
            className="label rounded-sm px-4 py-2.5 disabled:opacity-50"
            style={{ background: "var(--ink)", color: "var(--paper)" }}
          >
            {pending ? "Working" : "Ask"}
          </button>
        </div>
      </form>

      <div className="mb-10 flex flex-wrap gap-2">
        {PROBES.map((probe) => (
          <button
            key={probe.question}
            type="button"
            onClick={() => void run(probe.question)}
            className="rounded-sm border px-2.5 py-1.5 text-left text-xs"
            style={{ borderColor: "var(--rule)", color: "var(--muted)" }}
          >
            <span className="label block" style={{ color: "var(--amber)" }}>
              {probe.shows}
            </span>
            <span className="ev">{probe.question}</span>
          </button>
        ))}
      </div>

      {pending && (
        <div
          className="rounded-sm border px-3 py-2 text-sm"
          style={{ borderColor: "var(--rule)", color: "var(--muted)" }}
        >
          <p>
            Retrieving, then checking every claim against the section it cites. An answer
            takes about half a minute; the API sleeps when idle on a free plan, so a first
            question after a quiet spell waits about a minute more for it to wake.
          </p>
          {notice && (
            <p className="mt-2" style={{ color: "var(--amber)" }}>
              {notice}
            </p>
          )}
          <p className="mt-2 text-xs">
            Asking another question now replaces this one rather than queueing behind it.
          </p>
        </div>
      )}

      {error && (
        <p
          className="rounded-sm border px-3 py-2 text-sm"
          style={{ borderColor: "var(--rule)", color: "var(--contest)" }}
        >
          {error}
        </p>
      )}

      {result && (
        <>
          <AnswerCard result={result} />
          <TracePanel trace={result.trace} />
        </>
      )}

      {!result && !error && !pending && (
        <p className="text-sm" style={{ color: "var(--muted)" }}>
          Pick a question above, or ask your own. Every answer opens its working: what each
          retrieval arm found, what the temporal layer held back, and what the verifier struck out.
        </p>
      )}

      <footer className="mt-16 border-t pt-6 text-xs" style={{ borderColor: "var(--rule)", color: "var(--muted)" }}>
        <p>
          Read is autonomous; the system never writes to a source system. Retrieval is filtered by
          the asking user&rsquo;s access groups before the model sees anything.
        </p>
      </footer>
    </main>
  );
}
