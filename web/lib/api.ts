import type { AskResponse } from "./types";

const BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

/** Every group, so the demo shows behaviour rather than access control. */
export const DEMO_GROUPS = ["procurement", "brand_ops", "legal"];

const WAKE_ATTEMPTS = 4;
const WAKE_BACKOFF_MS = 6000;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function post(question: string): Promise<Response> {
  return fetch(`${BACKEND}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, user_groups: DEMO_GROUPS, user_id: "web" }),
  });
}

/**
 * Ask a question, absorbing a cold start rather than reporting it as a failure.
 *
 * On a free plan the API sleeps after fifteen minutes and takes about a minute to wake.
 * While it wakes, the connection is refused or answered with a 502 — so a single attempt
 * surfaces a waking service as a broken one. Retrying is the difference between a visitor
 * seeing a slow answer and seeing an error.
 */
export async function ask(question: string): Promise<AskResponse> {
  let lastReason = "";

  for (let attempt = 0; attempt < WAKE_ATTEMPTS; attempt++) {
    let response: Response;
    try {
      response = await post(question);
    } catch {
      // fetch rejects on a network-level failure with no detail available to us.
      lastReason = "the connection was refused";
      await sleep(WAKE_BACKOFF_MS);
      continue;
    }

    // 422 is the caller's fault and will not improve with a retry.
    if (response.status === 422) throw new Error("Enter a question first.");

    if (response.status === 502 || response.status === 503) {
      lastReason = `the service returned ${response.status}`;
      await sleep(WAKE_BACKOFF_MS);
      continue;
    }

    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`The API returned ${response.status}. ${detail.slice(0, 160)}`);
    }
    return response.json();
  }

  throw new Error(
    `Could not reach the API after ${WAKE_ATTEMPTS} attempts — ${lastReason}. It sleeps ` +
      "when idle on a free plan and takes about a minute to wake, so one more try often " +
      "succeeds. If it keeps failing, the service is down rather than asleep.",
  );
}
