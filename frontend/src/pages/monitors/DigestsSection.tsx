import { useMemo, useState } from "react";
import { Newspaper, Clock } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { markdownComponents } from "../../components/ChatMessage";
import { formatDate } from "../../lib/utils";
import type { MonitorInfo, MonitorResult } from "../../lib/types";
import { Card, EmptyState, Modal } from "../../components/ui";

function ResultChip({ status }: { status: string }) {
  const tone =
    status === "alert"
      ? "bg-nova-error/20 text-nova-error"
      : status === "changed"
        ? "bg-nova-warning/20 text-nova-warning"
        : "bg-nova-success/20 text-nova-success";
  return <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${tone}`}>{status}</span>;
}

// Same prose stack the chat + dossier readers use — digests are markdown.
const PROSE =
  "prose prose-sm max-w-none prose-p:my-1.5 prose-pre:bg-nova-bg prose-pre:border prose-pre:border-nova-border prose-pre:rounded-lg prose-code:text-nova-glow prose-a:text-nova-accent prose-a:no-underline hover:prose-a:underline prose-headings:text-nova-text prose-p:text-nova-text prose-li:text-nova-text prose-strong:text-nova-text prose-td:text-nova-text prose-th:text-nova-th prose-blockquote:text-nova-text-dim prose-blockquote:border-nova-accent/30";

interface Props {
  results: MonitorResult[];
  monitors: MonitorInfo[];
}

/** The reading room for what the monitors actually PRODUCE — the digests
 *  themselves, not their plumbing. Substantive results only (short status
 *  pings stay on the admin side). */
export default function DigestsSection({ results, monitors }: Props) {
  const [open, setOpen] = useState<MonitorResult | null>(null);

  const nameOf = useMemo(() => {
    const m = new Map<number, string>();
    monitors.forEach((x) => m.set(x.id, x.name));
    return m;
  }, [monitors]);

  const digests = useMemo(
    () =>
      results
        .filter((r) => (r.value?.length ?? 0) > 300)
        .sort((a, b) => (b.created_at > a.created_at ? 1 : -1)),
    [results]
  );

  if (digests.length === 0) {
    return (
      <EmptyState
        icon={<Newspaper className="h-8 w-8" />}
        title="No digests in the window"
        description="Substantive monitor output from the last 72 hours appears here as it lands."
      />
    );
  }

  // A real excerpt — the first few substantive lines, not just the opening
  // fragment (owner: "just truncated beginnings"). Full text opens on click.
  const excerpt = (v: string) => {
    const lines = v
      .split("\n")
      .map((l) => l.replace(/^[#*_>\s]+/, "").replace(/\*\*/g, "").trim())
      .filter((l) => l.length > 24);
    return lines.slice(0, 3).join(" ") || v.slice(0, 260);
  };

  return (
    <>
      <div className="space-y-2">
        {digests.map((d) => (
          <button key={d.id} onClick={() => setOpen(d)} className="w-full text-left">
            <Card className="transition hover:border-nova-accent/60">
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="truncate text-sm font-medium text-nova-text">
                  {nameOf.get(d.monitor_id) ?? `Monitor #${d.monitor_id}`}
                </span>
                <span className="flex shrink-0 items-center gap-2">
                  <ResultChip status={d.status} />
                  <span className="flex items-center gap-1 text-[11px] text-nova-text-dim">
                    <Clock className="h-3 w-3" /> {formatDate(d.created_at)}
                  </span>
                </span>
              </div>
              <p className="line-clamp-3 text-xs leading-relaxed text-nova-text-dim">{excerpt(d.value ?? "")}</p>
            </Card>
          </button>
        ))}
      </div>

      <Modal
        open={open !== null}
        onClose={() => setOpen(null)}
        title={open ? (nameOf.get(open.monitor_id) ?? "Digest") : ""}
      >
        {open && (
          <div className={`max-h-[70vh] overflow-y-auto pr-1 ${PROSE}`}>
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{open.value ?? ""}</ReactMarkdown>
          </div>
        )}
      </Modal>
    </>
  );
}
