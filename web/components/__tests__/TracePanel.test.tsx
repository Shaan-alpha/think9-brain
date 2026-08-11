import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { TracePanel } from "../TracePanel";
import type { Trace } from "@/lib/types";

const trace: Trace = {
  route: "factual_lookup",
  coverage: 0.9985,
  owner: "Priya Nair",
  retrieval: {
    dense: [
      { chunk_id: "c1", rank: 1, score: 0.81, snippet: "amber glass Rs 22.10" },
    ],
    sparse: [
      { chunk_id: "c2", rank: 1, score: 0.44, snippet: "Korent SKU AMB-50-FL" },
    ],
    fused: [{ chunk_id: "c1", rank: 1, score: 0.03, snippet: "amber glass" }],
    reranked: [{ chunk_id: "c1", rank: 1, score: 0.99, snippet: "amber glass" }],
    demoted: [
      {
        title: "korent-quote-2024-03.md",
        effective_date: "2024-03-12",
        demoted_by: "doc-2026",
      },
    ],
  },
  verifier: {
    stripped: ["Lead time is 91 days."],
    claims: [
      { claim: "Rs 22.10 per unit.", supported: true, reason: "entailed by evidence" },
      {
        claim: "Lead time is 91 days.",
        supported: false,
        reason: "ungrounded number '91'",
      },
    ],
  },
};

describe("TracePanel", () => {
  test("shows the router classification and coverage", async () => {
    render(<TracePanel trace={trace} />);
    await userEvent.click(screen.getByRole("button", { name: /trace/i }));
    expect(screen.getByTestId("route")).toHaveTextContent("factual_lookup");
    expect(screen.getByTestId("coverage")).toHaveTextContent("0.99");
  });

  test("shows dense and sparse candidates side by side", async () => {
    render(<TracePanel trace={trace} />);
    await userEvent.click(screen.getByRole("button", { name: /trace/i }));
    expect(screen.getByTestId("dense-column")).toHaveTextContent("Rs 22.10");
    expect(screen.getByTestId("sparse-column")).toHaveTextContent("AMB-50-FL");
  });

  test("names each demoted document and what superseded it", async () => {
    render(<TracePanel trace={trace} />);
    await userEvent.click(screen.getByRole("button", { name: /trace/i }));
    expect(screen.getByTestId("demotions")).toHaveTextContent(
      "korent-quote-2024-03.md",
    );
    expect(screen.getByTestId("demotions")).toHaveTextContent(/superseded/i);
  });

  test("lists claims the verifier stripped, with the reason", async () => {
    render(<TracePanel trace={trace} />);
    await userEvent.click(screen.getByRole("button", { name: /trace/i }));
    const verifier = screen.getByTestId("verifier");
    expect(verifier).toHaveTextContent("Lead time is 91 days.");
    expect(verifier).toHaveTextContent("ungrounded number");
  });

  test("is collapsed until opened", () => {
    render(<TracePanel trace={trace} />);
    expect(screen.queryByTestId("route")).not.toBeInTheDocument();
  });
});
