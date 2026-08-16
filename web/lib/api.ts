import type { AskResponse } from "./types";

const BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

/** Every group, so the demo shows behaviour rather than access control. */
export const DEMO_GROUPS = ["procurement", "brand_ops", "legal"];

/**
 * A warm answer takes about 25 seconds; a cold one waits for the free plan to wake first,
 * which the API's own logs put at 55–80 seconds. One attempt therefore has to be allowed
 * to run well past a minute before we can honestly call it stuck.
 */
const ATTEMPT_TIMEOUT_MS = 120_000;
const OVERALL_BUDGET_MS = 200_000;
const BACKOFF_MS = 4_000;

/**
 * Statuses worth trying again. 500 is on this list because of a real outage: the API held
 * one database connection for the life of the process, the database dropped it when idle,
 * and every question after that returned 500 until the container restarted. Retrying only
 * 502 and 503 meant the one failure that was actually happening was reported to the user
 * on the first try, which is why it looked permanently broken.
 */
const RETRYABLE = new Set([429, 500, 502, 503, 504]);

export class AskAborted extends Error {}

/** Wait between attempts, but give up immediately if a newer question has replaced this one. */
const backoff = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    if (signal?.aborted) return reject(new AskAborted("Superseded by a newer question."));
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new AskAborted("Superseded by a newer question."));
      },
      { once: true },
    );
  });

/**
 * One signal that fires on either the caller's cancellation or this attempt's timeout.
 *
 * Written out rather than using `AbortSignal.any`, which Safari only shipped in 17.4 — a
 * visitor on anything older would get a hard failure on every question instead of an
 * answer. `dispose` matters too: without it each attempt leaves a two-minute timer behind.
 */
function attemptSignal(caller: AbortSignal | undefined, timeoutMs: number) {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const relay = () => controller.abort();
  caller?.addEventListener("abort", relay, { once: true });

  return {
    signal: controller.signal,
    didTimeOut: () => timedOut,
    dispose: () => {
      clearTimeout(timer);
      caller?.removeEventListener("abort", relay);
    },
  };
}

export type AskOptions = {
  /** Cancels the request. A superseded question should not keep occupying the one instance. */
  signal?: AbortSignal;
  /** Called before each retry, so the UI can say "still waking" instead of just spinning. */
  onRetry?: (attempt: number, reason: string) => void;
};

async function post(question: string, signal: AbortSignal): Promise<Response> {
  return fetch(`${BACKEND}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, user_groups: DEMO_GROUPS, user_id: "web" }),
    signal,
  });
}

/**
 * Ask a question, absorbing a cold start rather than reporting it as a failure.
 *
 * On a free plan the API sleeps after fifteen minutes and takes about a minute to wake.
 * While it wakes, the connection is refused or answered with a 5xx — so a single attempt
 * surfaces a waking service as a broken one. Retrying is the difference between a visitor
 * seeing a slow answer and seeing an error.
 */
export async function ask(question: string, options: AskOptions = {}): Promise<AskResponse> {
  const { signal, onRetry } = options;
  const deadline = Date.now() + OVERALL_BUDGET_MS;
  let lastReason = "it did not respond";
  let attempt = 0;

  while (Date.now() < deadline) {
    attempt++;
    if (signal?.aborted) throw new AskAborted("Superseded by a newer question.");

    // Each attempt gets its own timeout so one hung connection cannot occupy the whole
    // budget, while the caller's signal still cancels immediately.
    const guard = attemptSignal(signal, Math.min(ATTEMPT_TIMEOUT_MS, deadline - Date.now()));

    let response: Response;
    try {
      response = await post(question, guard.signal);
    } catch {
      if (signal?.aborted) throw new AskAborted("Superseded by a newer question.");
      // fetch rejects on a network-level failure and on our own timeout, and gives us no
      // detail we could usefully show either way.
      lastReason = guard.didTimeOut()
        ? "it stopped responding mid-request"
        : "the connection was refused";
      onRetry?.(attempt, lastReason);
      await backoff(BACKOFF_MS, signal);
      continue;
    } finally {
      guard.dispose();
    }

    // 422 is the caller's fault and will not improve with a retry.
    if (response.status === 422) throw new Error("Enter a question first.");

    if (RETRYABLE.has(response.status)) {
      lastReason = `the service returned ${response.status}`;
      onRetry?.(attempt, lastReason);
      await backoff(BACKOFF_MS, signal);
      continue;
    }

    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`The API returned ${response.status}. ${detail.slice(0, 160)}`);
    }
    return response.json();
  }

  throw new Error(
    `Could not get an answer after ${attempt} attempts over ${Math.round(
      OVERALL_BUDGET_MS / 1000,
    )} seconds — ${lastReason}. The API sleeps when idle on a free plan and takes about a ` +
      "minute to wake, so one more try often succeeds. If it keeps failing, the service is " +
      "down rather than asleep.",
  );
}
