import type { StreamEvent, EventType } from "./types";

export interface SSEOptions {
  onEvent: (event: StreamEvent) => void;
  onError: (error: Error) => void;
  signal?: AbortSignal;
  imageBase64?: string;
}

/**
 * POST-based SSE parser. Native EventSource doesn't support POST,
 * so we use fetch + ReadableStream with manual event: / data: parsing.
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

  let res: Response;
  try {
    res = await fetch("/api/chat/stream", {
      method: "POST",
      headers,
      body,
      signal: options.signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    options.onError(err as Error);
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
    return null;
  }
}
