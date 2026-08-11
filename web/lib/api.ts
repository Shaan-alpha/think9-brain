import type { AskResponse } from "./types";

const BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

/** Every group, so the demo shows behaviour rather than access control. */
export const DEMO_GROUPS = ["procurement", "brand_ops", "legal"];

export async function ask(question: string): Promise<AskResponse> {
  let response: Response;
  try {
    response = await fetch(`${BACKEND}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        user_groups: DEMO_GROUPS,
        user_id: "web",
      }),
    });
  } catch {
    // fetch only rejects on a network-level failure, where the browser gives us no
    // detail. "Failed to fetch" on its own tells a reader nothing about what to do.
    throw new Error(
      "Could not reach the API. It sleeps when idle on a free plan — wait a few seconds " +
        "and ask again. If it keeps failing, the service is down rather than asleep.",
    );
  }

  if (response.status === 422) throw new Error("Enter a question first.");
  if (response.status === 502 || response.status === 503) {
    throw new Error(
      "The API is starting up. It loads two models into memory on boot, which takes " +
        "about a minute from cold. Ask again shortly.",
    );
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`The API returned ${response.status}. ${detail.slice(0, 160)}`);
  }
  return response.json();
}
