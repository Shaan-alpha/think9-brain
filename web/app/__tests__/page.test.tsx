import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import Home from "../page";
import type { AskResponse } from "@/lib/types";

const { askMock } = vi.hoisted(() => ({ askMock: vi.fn() }));

vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  ask: askMock,
}));

const answerFor = (text: string): AskResponse => ({
  answer: text,
  outcome: "answered",
  as_of: "2026-01-08",
  citations: [],
  trace: {},
});

/** A promise plus the handle to settle it, so a test controls when each request finishes. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

describe("Home", () => {
  beforeEach(() => askMock.mockReset());
  afterEach(() => vi.restoreAllMocks());

  test("a second question replaces the first instead of racing it", async () => {
    // Both probes used to stay clickable mid-request, leaving two questions in flight on
    // one small instance. Whichever finished last won the render, so the answer on screen
    // could belong to a question the visitor had already moved on from.
    const first = deferred<AskResponse>();
    const second = deferred<AskResponse>();
    askMock
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise);

    const user = userEvent.setup();
    render(<Home />);

    await user.click(screen.getByText("What do we pay for 50ml amber glass?"));
    await user.click(screen.getByText("Why did we discontinue the mango variant?"));

    // The superseded request lands last, which is the ordering that used to win.
    second.resolve(answerFor("Lowest purchase intent of six candidates."));
    first.resolve(answerFor("Rs 22.10 per unit."));

    await waitFor(() =>
      expect(screen.getByText(/Lowest purchase intent/)).toBeInTheDocument(),
    );
    expect(screen.queryByText(/Rs 22.10 per unit/)).not.toBeInTheDocument();
  });

  test("the superseded request is actually cancelled, not just ignored", async () => {
    // Leaving it running would keep competing for the single instance that the new
    // question now needs.
    const first = deferred<AskResponse>();
    const second = deferred<AskResponse>();
    askMock
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise);

    const user = userEvent.setup();
    render(<Home />);

    await user.click(screen.getByText("What do we pay for 50ml amber glass?"));
    const firstSignal = askMock.mock.calls[0][1].signal as AbortSignal;
    expect(firstSignal.aborted).toBe(false);

    await user.click(screen.getByText("Why did we discontinue the mango variant?"));
    expect(firstSignal.aborted).toBe(true);

    // Settle both so nothing is left pending once the component unmounts.
    first.resolve(answerFor("superseded"));
    second.resolve(answerFor("Lowest purchase intent of six candidates."));
    await waitFor(() =>
      expect(screen.getByText(/Lowest purchase intent/)).toBeInTheDocument(),
    );
  });

  test("cancelling does not show the visitor an error", async () => {
    const { AskAborted } = await import("@/lib/api");
    askMock
      .mockImplementationOnce(() => Promise.reject(new AskAborted("superseded")))
      .mockImplementationOnce(() => Promise.resolve(answerFor("Rs 22.10 per unit.")));

    const user = userEvent.setup();
    render(<Home />);

    await user.click(screen.getByText("What do we pay for 50ml amber glass?"));
    await user.click(screen.getByText("Why did we discontinue the mango variant?"));

    await waitFor(() => expect(screen.getByText(/Rs 22.10 per unit/)).toBeInTheDocument());
  });
});
