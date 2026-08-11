import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { AnswerCard } from "../AnswerCard";
import type { AskResponse } from "@/lib/types";

const answered: AskResponse = {
  answer: "We pay Rs 22.10 per unit for a 50ml amber glass jar [c:abc].",
  outcome: "answered",
  as_of: "2026-01-08",
  citations: [
    {
      chunk_id: "abc",
      document_title: "korent-quote-2026-01.md",
      heading_path: "Pricing",
      deep_link: "https://drive/f1",
      effective_date: "2026-01-08",
    },
  ],
  trace: {},
};

describe("AnswerCard", () => {
  test("shows the as-of date badge", () => {
    render(<AnswerCard result={answered} />);
    expect(screen.getByTestId("as-of")).toHaveTextContent("8 Jan 2026");
  });

  test("renders each citation as a link to the original document", () => {
    render(<AnswerCard result={answered} />);
    const link = screen.getByRole("link", { name: /korent-quote-2026-01/ });
    expect(link).toHaveAttribute("href", "https://drive/f1");
  });

  test("replaces raw citation markers with numbered references", () => {
    render(<AnswerCard result={answered} />);
    expect(screen.getByTestId("answer-text")).not.toHaveTextContent("[c:abc]");
    expect(screen.getByTestId("answer-text")).toHaveTextContent("Rs 22.10");
  });

  test("a refusal is marked and carries no as-of badge", () => {
    render(
      <AnswerCard
        result={{ ...answered, outcome: "refused", as_of: null, citations: [] }}
      />,
    );
    expect(screen.getByTestId("outcome")).toHaveTextContent(/refused/i);
    expect(screen.queryByTestId("as-of")).not.toBeInTheDocument();
  });

  test("a contested answer shows both values side by side", () => {
    render(
      <AnswerCard
        result={{
          ...answered,
          outcome: "contested",
          answer: "Two current sources disagree.",
          trace: {
            contested: {
              attribute: "minimum order quantity",
              values: [
                ["5,000", "korent-spec-sheet-2025-11.md"],
                ["8,000", "korent-contract-annexe-2025-12.md"],
              ],
            },
          },
        }}
      />,
    );
    expect(screen.getByTestId("contested")).toHaveTextContent("5,000");
    expect(screen.getByTestId("contested")).toHaveTextContent("8,000");
    expect(screen.getByTestId("contested")).toHaveTextContent(
      /minimum order quantity/,
    );
  });
});
