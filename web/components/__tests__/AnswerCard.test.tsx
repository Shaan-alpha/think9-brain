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

  test("a reference hugs the word it follows rather than floating before the full stop", () => {
    /* The model writes "...scored lowest [c:abc]." If that space survives, the marker
       renders adrift between the claim and its punctuation: "lowest 1. Panellists". */
    render(
      <AnswerCard
        result={{ ...answered, answer: "It scored lowest [c:abc]. Panellists agreed." }}
      />,
    );

    expect(screen.getByTestId("answer-text")).toHaveTextContent("lowest1. Panellists agreed.");
  });

  test("two references in a row stay separate rather than reading as one number", () => {
    render(
      <AnswerCard
        result={{
          ...answered,
          answer: "Both sources agree [c:abc][c:def].",
          citations: [
            ...answered.citations,
            {
              chunk_id: "def",
              document_title: "grove-consumer-panel-2025-05.md",
              heading_path: "Mango variant",
              deep_link: "https://drive/f2",
              effective_date: "2025-05-14",
            },
          ],
        }}
      />,
    );

    // A CSS margin would look right and still copy as "agree12", so the separator has to
    // be a real character.
    expect(screen.getByTestId("answer-text").textContent).toContain("agree1,2.");
  });

  test("a long filename and heading path both survive intact", () => {
    /* The values that broke the layout. The tests used `korent-quote-2026-01.md` under
       the heading `Pricing`, which is short enough to fit beside its metadata — so the
       filename being crushed to a six-line column never showed up here. Real citations
       look like this. */
    render(
      <AnswerCard
        result={{
          ...answered,
          citations: [
            {
              chunk_id: "abc",
              document_title: "grove-mango-decision-memo-2025-07.md",
              heading_path: "Decision Memo — Discontinuing the Grove Mango Variant > Reasoning",
              deep_link: "https://drive/f9",
              effective_date: "2025-07-01",
            },
          ],
        }}
      />,
    );

    // One link whose text is the whole filename, not a fragment of it.
    const link = screen.getByRole("link", { name: "grove-mango-decision-memo-2025-07.md" });
    expect(link).toHaveAttribute("href", "https://drive/f9");

    // The heading path is where the answer is grounded and the date is what makes it
    // current, so both are shown at every width rather than hidden on small screens.
    const source = link.closest("li");
    expect(source).toHaveTextContent("Decision Memo — Discontinuing the Grove Mango Variant");
    expect(source).toHaveTextContent("2025-07-01");
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
