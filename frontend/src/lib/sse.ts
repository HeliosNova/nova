import type { StreamEvent, EventType } from "./types";
import { getBaseUrl } from "./api";

export interface SSEOptions {
  onEvent: (event: StreamEvent) => void;
  onError: (error: Error) => void;
  signal?: AbortSignal;
  imageBase64?: string;
}

const MAX_RETRIES = 3;
const BASE_DELAY_MS = 1000;

/** Returns true if the error/status is retryable (network error or 502/503). */
function isRetryable(err: unknown, status?: number): boolean {
  if (status === 502 || status === 503) return true;
  // TypeError from fetch indicates a network-level failure
  if (err instanceof TypeError) return true;
  return false;
}

/** Sleep helper that resolves early if the signal is aborted. */
function delay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => { clearTimeout(timer); resolve(); }, { once: true });
  });
}

/**
 * POST-based SSE parser. Native EventSource doesn't support POST,
 * so we use fetch + ReadableStream with manual event: / data: parsing.
 *
 * Retries up to MAX_RETRIES times with exponential backoff for network
 * errors and 502/503 responses. Does NOT retry on 4xx client errors.
 */
export async function streamChat(
  query: string,
  conversationId: string | null,
  options: SSEOptions
): Promise<void> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  const apiKey = localStorage.getItem("nova_api_key");
  if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

  const payload: Record<string, unknown> = {
    query,
    conversation_id: conversationId,
  };
  if (options.imageBase64) {
    payload.image_base64 = options.imageBase64;
  }
  const body = JSON.stringify(payload);

  let res: Response | undefined;
  let lastError: Error | undefined;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    if (options.signal?.aborted) return;

    try {
      res = await fetch(`${getBaseUrl()}/api/chat/stream`, {
        method: "POST",
        headers,
        body,
        signal: options.signal,
      });
    } catch (err) {
      if ((err as Error).name === "AbortError") return;

      if (isRetryable(err) && attempt < MAX_RETRIES) {
        lastError = err as Error;
        await delay(BASE_DELAY_MS * 2 ** attempt, options.signal);
        continue;
      }

      options.onError(err as Error);
      return;
    }

    // Retry on 502/503, but NOT on other non-OK statuses
    if ((res.status === 502 || res.status === 503) && attempt < MAX_RETRIES) {
      lastError = new Error(`${res.status}: ${res.statusText}`);
      await delay(BASE_DELAY_MS * 2 ** attempt, options.signal);
      continue;
    }

    // Success or non-retryable error — break out of retry loop
    break;
  }

  if (!res) {
    options.onError(lastError ?? new Error("Failed to connect after retries"));
    return;
  }

  if (!res.ok) {
    const text = await res.text();
    options.onError(new Error(`${res.status}: ${text}`));
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    options.onError(new Error("No readable stream"));
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE messages are separated by double newlines
      const parts = buffer.split("\n\n");
      // Last part may be incomplete
      buffer = parts.pop() || "";

      for (const part of parts) {
        const event = parseSSEBlock(part.trim());
        if (event) {
          options.onEvent(event);
        }
      }
    }

    // Process any remaining buffer
    if (buffer.trim()) {
      const event = parseSSEBlock(buffer.trim());
      if (event) {
        options.onEvent(event);
      }
    }
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    options.onError(err as Error);
  }
}

function parseSSEBlock(block: string): StreamEvent | null {
  let eventType: string | null = null;
  const dataLines: string[] = [];

  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) {
      eventType = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }

  const dataStr = dataLines.join("\n");

  if (!eventType || !dataStr) return null;

  try {
    const data = JSON.parse(dataStr);
    return { type: eventType as EventType, data };
  } catch {
    if (import.meta.env.DEV) {
      console.warn("[SSE] Failed to parse event data:", eventType, dataStr.slice(0, 200));
    }
    return null;
  }
}
