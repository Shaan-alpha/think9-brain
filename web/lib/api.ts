import type { AskResponse } from "./types";

const BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

/** Every group, so the demo shows behaviour rather than access control. */
export const DEMO_GROUPS = ["procurement", "brand_ops", "legal"];

export async function ask(question: string): Promise<AskResponse> {
  const response = await fetch(`${BACKEND}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      user_groups: DEMO_GROUPS,
      user_id: "web",
    }),
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(
      response.status === 422
        ? "Enter a question first."
        : `The Brain is unreachable (${response.status}). ${detail.slice(0, 120)}`,
    );
  }
  return response.json();
}
