import { useMemo, useState } from "react";
import { ArrowLeft } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { markdownComponents } from "../components/ChatMessage";
import { formatDate, parseTs } from "../lib/utils";
import type { MonitorInfo, MonitorResult } from "../lib/types";

const PROSE =
  "prose prose-sm max-w-none prose-p:my-1.5 prose-headings:text-nova-text prose-headings:font-display prose-p:text-nova-text prose-li:text-nova-text prose-strong:text-nova-glow prose-a:text-nova-accent prose-a:no-underline hover:prose-a:underline prose-blockquote:text-nova-text-dim prose-blockquote:border-nova-accent/30 prose-hr:border-nova-border";

const clean = (v: string) =>
  v.split("\n").map((l) => l.replace(/^[#*_>\s·•📝📊🕵️💡💵🟢●↳]+/u, "").replace(/\*\*/g, "").trim()).filter((l) => l.length > 22);

/** Signals region: the monitor intelligence as an editorial briefing desk —
 *  wire entries (desk · headline · lede · time), hairline-divided. Clicking a
 *  brief opens it in-panel as a full reader (same as Dossiers), not a modal. */
const PAGE = 30;

export default function SignalsBriefing({ results, monitors, loading }: { results: MonitorResult[]; monitors: MonitorInfo[]; loading?: boolean }) {
  const [open, setOpen] = useState<MonitorResult | null>(null);
  const [limit, setLimit] = useState(PAGE);
  const nameOf = useMemo(() => { const m = new Map<number, string>(); monitors.forEach((x) => m.set(x.id, x.name)); return m; }, [monitors]);
  const digests = useMemo(
    () => results.filter((r) => (r.value?.length ?? 0) > 300).sort((a, b) => parseTs(b.created_at) - parseTs(a.created_at)),
    [results]
  );

  // in-panel reader (mirrors DossiersPanel)
  if (open) {
    return (
      <div className="p-4 sm:p-5">
        <button onClick={() => setOpen(null)} className="mb-3 inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.16em] text-nova-text-dim transition-colors hover:text-nova-accent">
          <ArrowLeft size={12} /> All briefings
        </button>
        <div className="mb-1 font-mono text-[9px] uppercase tracking-[0.24em] text-nova-text-dim/80">{open.status} · {formatDate(open.created_at)}</div>
        <h3 className="mb-3 font-display text-2xl font-semibold tracking-[-0.02em] text-nova-text">{nameOf.get(open.monitor_id) ?? "Briefing"}</h3>
        <div className={PROSE}><ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{open.value ?? ""}</ReactMarkdown></div>
      </div>
    );
  }

  if (digests.length === 0) {
    if (loading) return <div className="space-y-2 p-4 sm:p-5">{[0, 1, 2, 3].map((i) => <div key={i} className="h-20 animate-pulse rounded-lg bg-nova-surface/40" />)}</div>;
    return <div className="p-8 text-center text-sm text-nova-text-dim">No briefings in the last 72 hours.</div>;
  }

  return (
    <div className="p-4 sm:p-5">
      <div className="mb-4 font-mono text-[10px] uppercase tracking-[0.22em] text-nova-text-dim">
        Intelligence desk · {digests.length} briefings · 72h window
      </div>
      <div className="space-y-2">
        {digests.slice(0, limit).map((d) => {
          const lines = clean(d.value ?? "");
          return (
            <button key={d.id} onClick={() => setOpen(d)}
              className="group w-full rounded-lg border border-nova-border bg-nova-surface/50 p-3.5 text-left backdrop-blur-md transition-colors hover:border-nova-accent/40">
              <div className="flex items-start justify-between gap-3">
                <span className="flex items-center gap-2 font-display text-[15px] font-medium leading-snug text-nova-text group-hover:text-nova-accent">
                  {d.status === "alert" && <span className="mt-1.5 h-1.5 w-1.5 shrink-0 self-start rounded-full bg-nova-error" />}
                  {d.status === "changed" && <span className="mt-1.5 h-1.5 w-1.5 shrink-0 self-start rounded-full bg-nova-warning" />}
                  {lines[0] ?? "Briefing"}
                </span>
                <span className="shrink-0 font-mono text-[9px] tabular-nums text-nova-text-dim/70">{formatDate(d.created_at)}</span>
              </div>
              {lines[1] && <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-nova-text-dim">{lines.slice(1, 3).join(" ")}</p>}
              <div className="mt-1.5 font-mono text-[9px] uppercase tracking-[0.14em] text-nova-accent/80">{nameOf.get(d.monitor_id) ?? `Monitor #${d.monitor_id}`}</div>
            </button>
          );
        })}
      </div>
      {digests.length > limit && (
        <button onClick={() => setLimit((n) => n + PAGE)}
          className="mt-3 w-full rounded-lg border border-nova-border bg-nova-surface/40 py-2 font-mono text-[10px] uppercase tracking-[0.14em] text-nova-text-dim transition-colors hover:border-nova-accent/40 hover:text-nova-text">
          Show {Math.min(PAGE, digests.length - limit)} more · {digests.length - limit} remaining
        </button>
      )}
    </div>
  );
}
