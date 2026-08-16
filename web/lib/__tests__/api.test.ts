import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { AskAborted, ask } from "../api";
import type { AskResponse } from "../types";

const ANSWER: AskResponse = {
  answer: "Rs 22.10 per unit.",
  outcome: "answered",
  as_of: "2026-01-08",
  citations: [],
  trace: {},
};

const ok = () => new Response(JSON.stringify(ANSWER), { status: 200 });
const failing = (status: number) => new Response("boom", { status });

/** Backoff is real time in the implementation, so the clock is faked for every test. */
async function drain() {
  await vi.advanceTimersByTimeAsync(10_000);
}

describe("ask", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  test("retries a 500 and returns the answer from the next attempt", async () => {
    // The outage that made the app look permanently broken: the API held one database
    // connection, the database dropped it when idle, and every question returned 500.
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(failing(500))
      .mockResolvedValueOnce(ok());
    vi.stubGlobal("fetch", fetchMock);

    const pending = ask("what do we pay for amber glass?");
    await drain();

    await expect(pending).resolves.toMatchObject({ outcome: "answered" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  test.each([502, 503, 504, 429])("retries a %i", async (status) => {
    const fetchMock = vi.fn().mockResolvedValueOnce(failing(status)).mockResolvedValueOnce(ok());
    vi.stubGlobal("fetch", fetchMock);

    const pending = ask("q");
    await drain();

    await expect(pending).resolves.toMatchObject({ outcome: "answered" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  test("retries a refused connection, which is what a waking service looks like", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(ok());
    vi.stubGlobal("fetch", fetchMock);

    const pending = ask("q");
    await drain();

    await expect(pending).resolves.toMatchObject({ outcome: "answered" });
  });

  test("does not retry a 422, because the question itself is the problem", async () => {
    const fetchMock = vi.fn().mockResolvedValue(failing(422));
    vi.stubGlobal("fetch", fetchMock);

    const pending = ask("   ");
    await expect(pending).rejects.toThrow("Enter a question first.");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test("reports each retry so the UI can say waking rather than just spinning", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValueOnce(failing(502)).mockResolvedValueOnce(ok()),
    );
    const onRetry = vi.fn();

    const pending = ask("q", { onRetry });
    await drain();
    await pending;

    expect(onRetry).toHaveBeenCalledWith(1, "the service returned 502");
  });

  test("a cancelled request raises AskAborted so the page can ignore it", async () => {
    const controller = new AbortController();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(
        (_url: string, init: RequestInit) =>
          new Promise((_resolve, reject) => {
            init.signal?.addEventListener("abort", () =>
              reject(new DOMException("Aborted", "AbortError")),
            );
          }),
      ),
    );

    const pending = ask("superseded question", { signal: controller.signal });
    controller.abort();

    await expect(pending).rejects.toBeInstanceOf(AskAborted);
  });
});
