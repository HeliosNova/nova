/**
 * Merge Tailwind class names, filtering out falsy values.
 */
export function cn(...classes: (string | false | null | undefined)[]): string {
  return classes.filter(Boolean).join(" ");
}

/**
 * Format an ISO date string to a human-friendly time.
 * < 1 min: "just now", < 60 min: "5m ago", today: "4:30 PM",
 * yesterday: "Yesterday 4:30 PM", older: "Mar 25, 4:30 PM"
 */
export function formatDate(iso: string): string {
  const date = new Date(iso);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMin = Math.floor(diffMs / 60_000);

  if (diffMin < 1) return "just now";
  if (diffMin < 60) return `${diffMin}m ago`;

  const time = date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  if (now.toDateString() === date.toDateString()) return time;
  const yesterday = new Date(now.getTime() - 86_400_000);
  if (yesterday.toDateString() === date.toDateString()) return `Yesterday ${time}`;
  return date.toLocaleDateString([], { month: "short", day: "numeric" }) + `, ${time}`;
}

/**
 * Truncate a string to a max length, appending "..." if truncated.
 */
export function truncate(str: string, max: number): string {
  if (str.length <= max) return str;
  return str.slice(0, max - 3) + "...";
}

/**
 * Generate a short unique ID for tool call tracking.
 */
export function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

/**
 * Format a number as a percentage string.
 */
export function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/**
 * Format bytes into a human-readable file size string.
 */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

/**
 * Format seconds into a human-readable duration string.
 */
export function formatSeconds(seconds: number): string {
  if (seconds < 60) return `${seconds} second${seconds !== 1 ? "s" : ""}`;
  if (seconds < 3600) {
    const m = Math.round(seconds / 60);
    return `${m} minute${m !== 1 ? "s" : ""}`;
  }
  if (seconds < 86400) {
    const h = Math.round(seconds / 3600);
    return `${h} hour${h !== 1 ? "s" : ""}`;
  }
  const d = Math.round(seconds / 86400);
  return `${d} day${d !== 1 ? "s" : ""}`;
}
