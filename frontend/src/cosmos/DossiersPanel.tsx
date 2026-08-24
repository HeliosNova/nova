import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, BookOpen } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { markdownComponents } from "../components/ChatMessage";
import { getDossiers, getDossier } from "../lib/api";
import { formatDate, parseTs } from "../lib/utils";
import type { DossierSummary, DossierDetail } from "../lib/types";

const PROSE =
  "prose prose-sm max-w-none prose-p:my-1.5 prose-headings:text-nova-text prose-headings:font-display prose-p:text-nova-text prose-li:text-nova-text prose-strong:text-nova-glow prose-a:text-nova-accent prose-blockquote:text-nova-text-dim prose-blockquote:border-nova-accent/30";

const KINDS: { id: string; label: string }[] = [
  { id: "all", label: "All" },
  { id: "meta", label: "Meta" },
  { id: "domain", label: "Domains" },
  { id: "entity", label: "Entities" },
  { id: "storyline", label: "Threads" },
];

const PAGE = 30;

/** Nova's living understanding — the dossiers it revises over time. */
export default function DossiersPanel() {
  const [items, setItems] = useState<DossierSummary[]>([]);
  const [kind, setKind] = useState("all");
  const [limit, setLimit] = useState(PAGE);
  const [open, setOpen] = useState<DossierDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [listLoading, setListLoading] = useState(true);

  useEffect(() => {
    getDossiers().then((v) => setItems(Array.isArray(v) ? v : [])).catch(() => setItems([])).finally(() => setListLoading(false));
  }, []);

  const shown = useMemo(
    () => (kind === "all" ? items : items.filter((d) => d.kind === kind)).slice().sort((a, b) => parseTs(b.updated_at) - parseTs(a.updated_at)),
    [items, kind]
  );

  const read = (id: number) => { setLoading(true); getDossier(id).then(setOpen).catch(() => {}).finally(() => setLoading(false)); };

  if (open) {
    return (
      <div>
        <button onClick={() => setOpen(null)} className="mb-3 inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.16em] text-nova-text-dim transition-colors hover:text-nova-accent">
          <ArrowLeft size={12} /> All dossiers
        </button>
        <div className="mb-1 font-mono text-[9px] uppercase tracking-[0.24em] text-nova-text-dim/70">{open.kind} · rev {open.revision_count} · {formatDate(open.updated_at)}</div>
        <h3 className="mb-3 font-display text-2xl font-semibold tracking-[-0.02em] text-nova-text">{open.title}</h3>
        <div className={PROSE}><ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{open.body || ""}</ReactMarkdown></div>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-1.5">
        {KINDS.map((k) => (
          <button key={k.id} onClick={() => { setKind(k.id); setLimit(PAGE); }}
            className={`rounded-md border px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.12em] transition-colors ${kind === k.id ? "border-nova-accent/40 bg-nova-accent/10 text-nova-accent" : "border-nova-border bg-nova-surface/40 text-nova-text-dim hover:text-nova-text"}`}>
            {k.label}
          </button>
        ))}
      </div>
      {(loading || listLoading) && <div className="space-y-2">{[0, 1, 2].map((i) => <div key={i} className="h-[68px] animate-pulse rounded-lg bg-nova-surface/40" />)}</div>}
      <div className="space-y-2">
        {shown.slice(0, limit).map((d) => (
          <button key={d.id} onClick={() => read(d.id)}
            className="group w-full rounded-lg border border-nova-border bg-nova-surface/50 p-3.5 text-left backdrop-blur-md transition-colors hover:border-nova-accent/40">
            <div className="flex items-start justify-between gap-3">
              <span className="font-display text-[15px] font-medium text-nova-text group-hover:text-nova-accent">{d.title}</span>
              <span className="shrink-0 font-mono text-[9px] uppercase tracking-[0.14em] text-nova-text-dim/70">{d.kind}</span>
            </div>
            {d.changed_note && <p className="mt-1 line-clamp-2 text-xs text-nova-text-dim">{d.changed_note}</p>}
            <div className="mt-1.5 font-mono text-[9px] tracking-tight text-nova-text-dim/60">{d.update_count} updates · {formatDate(d.updated_at)}</div>
          </button>
        ))}
        {shown.length === 0 && !loading && !listLoading && (
          <div className="flex flex-col items-center gap-2 py-12 text-center text-nova-text-dim">
            <BookOpen size={22} className="opacity-50" />
            <p className="text-sm">No dossiers of this kind yet.</p>
          </div>
        )}
      </div>
      {shown.length > limit && (
        <button onClick={() => setLimit((n) => n + PAGE)}
          className="mt-3 w-full rounded-lg border border-nova-border bg-nova-surface/40 py-2 font-mono text-[10px] uppercase tracking-[0.14em] text-nova-text-dim transition-colors hover:border-nova-accent/40 hover:text-nova-text">
          Show {Math.min(PAGE, shown.length - limit)} more · {shown.length - limit} remaining
        </button>
      )}
    </div>
  );
}
