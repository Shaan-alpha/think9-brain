"use client";

import { useState } from "react";
import { AnswerCard } from "@/components/AnswerCard";
import { TracePanel } from "@/components/TracePanel";
import { ask } from "@/lib/api";
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

  async function run(q: string) {
    if (!q.trim()) return;
    setQuestion(q);
    setPending(true);
    setError(null);
    try {
      setResult(await ask(q));
    } catch (e) {
      setResult(null);
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setPending(false);
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
