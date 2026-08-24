import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, LayoutGrid, List, ExternalLink, GitBranch } from "lucide-react";
import { getStorylines, getStoryline } from "../lib/api";
import { formatDate, parseTs } from "../lib/utils";
import type { StorylineSummary, StorylineEvent } from "../lib/types";

const FILTERS = [
  { id: "active", label: "Active" },
  { id: "closed", label: "Closed" },
  { id: "all", label: "All" },
] as const;
type Filter = (typeof FILTERS)[number]["id"];

/** Storylines region: Nova's narrative threads, organized like the dossiers — a
 *  calm card list that opens each thread in-panel as a full reader (summary +
 *  event timeline). A Map view toggles to the swimlane thread-map for the
 *  time-shape at a glance; its dots open the same reader. */
export default function StorylinesMap() {
  const [items, setItems] = useState<StorylineSummary[]>([]);
  const [stats, setStats] = useState<{ active: number; closed: number } | null>(null);
  const [details, setDetails] = useState<Record<number, StorylineEvent[]>>({});
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<Filter>("active");
  const [view, setView] = useState<"list" | "map">("list");
  const [limit, setLimit] = useState(30);
  const [open, setOpen] = useState<StorylineSummary | null>(null);
  const [hover, setHover] = useState<{ e: StorylineEvent; title: string } | null>(null);

  useEffect(() => {
    getStorylines("all", 60)
      .then((d) => { setItems(Array.isArray(d.storylines) ? d.storylines : []); setStats(d.stats); })
      .catch(() => setItems([]))
      .finally(() => setLoading(false));
  }, []);

  const shown = useMemo(
    () => items.filter((s) => filter === "all" || s.status === filter)
               .slice().sort((a, b) => parseTs(b.last_updated) - parseTs(a.last_updated)),
    [items, filter]
  );

  // load a thread's events on demand (reader) or in bulk (map) — cached by id
  const ensureDetail = (id: number) => {
    if (details[id]) return;
    getStoryline(id).then((x) => setDetails((m) => ({ ...m, [id]: x.events }))).catch(() => {});
  };
  const read = (s: StorylineSummary) => { ensureDetail(s.id); setOpen(s); };

  // the map draws the top threads by recency; pull their events when it's shown
  const mapThreads = useMemo(() => shown.slice(0, 14), [shown]);
  useEffect(() => { if (view === "map") mapThreads.forEach((s) => ensureDetail(s.id)); }, [view, mapThreads]); // eslint-disable-line react-hooks/exhaustive-deps
  const mapRows = useMemo(() => mapThreads.map((s) => ({ s, events: details[s.id] ?? [] })), [mapThreads, details]);

  if (loading) return <div className="p-4 sm:p-5"><div className="h-64 animate-pulse rounded-lg bg-nova-surface/40" /></div>;

  // ---- in-panel reader (mirrors DossiersPanel) ----
  if (open) {
    const loaded = open.id in details;
    const events = details[open.id] ?? [];
    return (
      <div className="p-4 sm:p-5">
        <button onClick={() => setOpen(null)} className="mb-3 inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.16em] text-nova-text-dim transition-colors hover:text-nova-accent">
          <ArrowLeft size={12} /> All storylines
        </button>
        <div className="mb-1 font-mono text-[9px] uppercase tracking-[0.24em] text-nova-text-dim/70">
          <span className={open.status === "active" ? "text-nova-success" : ""}>{open.status}</span> · {open.event_count} events · updated {formatDate(open.last_updated)}
        </div>
        <h3 className="mb-2 font-display text-2xl font-semibold tracking-[-0.02em] text-nova-text">{open.title}</h3>
        {open.summary && <p className="mb-5 text-sm leading-relaxed text-nova-text/85">{open.summary}</p>}
        <div className="mb-2.5 font-mono text-[9px] uppercase tracking-[0.2em] text-nova-text-dim/70">Timeline · {events.length} updates</div>
        <ol className="relative space-y-3.5 border-l border-nova-border/60 pl-4">
          {events.map((e) => (
            <li key={e.id} className="relative">
              <span className={`absolute -left-[21px] top-1 h-2.5 w-2.5 rounded-full ring-2 ring-nova-bg ${e.is_new ? "bg-nova-accent" : "bg-nova-border-bright"}`} />
              <p className="whitespace-pre-line text-[13px] leading-relaxed text-nova-text/90">{e.summary}</p>
              <div className="mt-1 flex flex-wrap items-center gap-x-2.5 font-mono text-[9px] tracking-tight text-nova-text-dim/70">
                <span>{formatDate(e.created_at)}</span>
                {e.source && <span className="max-w-[220px] truncate">{e.source}</span>}
                {e.url && <a href={e.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-0.5 text-nova-accent/80 hover:text-nova-accent">source <ExternalLink size={8} /></a>}
              </div>
            </li>
          ))}
          {events.length === 0 && <p className="text-xs text-nova-text-dim/70">{loaded ? "No timeline events recorded yet." : "Loading timeline…"}</p>}
        </ol>
      </div>
    );
  }

  // ---- list + map ----
  return (
    <div className="p-4 sm:p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-1.5">
          {FILTERS.map((f) => (
            <button key={f.id} onClick={() => { setFilter(f.id); setLimit(30); }}
              className={`rounded-md border px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.12em] transition-colors ${filter === f.id ? "border-nova-accent/40 bg-nova-accent/10 text-nova-accent" : "border-nova-border bg-nova-surface/40 text-nova-text-dim hover:text-nova-text"}`}>
              {f.label}{stats && f.id !== "all" ? <span className="ml-1.5 tabular-nums text-nova-text-dim/70">{f.id === "active" ? stats.active : stats.closed}</span> : null}
            </button>
          ))}
        </div>
        <div className="flex gap-1">
          {([["list", List, "Text"], ["map", LayoutGrid, "Map"]] as const).map(([v, Icon, lbl]) => (
            <button key={v} onClick={() => setView(v)}
              className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-[0.12em] transition-colors ${view === v ? "border-nova-accent/40 bg-nova-accent/10 text-nova-accent" : "border-nova-border bg-nova-surface/40 text-nova-text-dim hover:text-nova-text"}`}>
              <Icon size={12} /> {lbl}
            </button>
          ))}
        </div>
      </div>

      {shown.length === 0 ? (
        <div className="flex flex-col items-center gap-2 py-12 text-center text-nova-text-dim">
          <GitBranch size={22} className="opacity-50" />
          <p className="text-sm">No {filter === "all" ? "" : filter} storylines yet.</p>
        </div>
      ) : view === "list" ? (
        <div className="space-y-2">
          {shown.slice(0, limit).map((s) => (
            <button key={s.id} onClick={() => read(s)}
              className="group w-full rounded-lg border border-nova-border bg-nova-surface/50 p-3.5 text-left backdrop-blur-md transition-colors hover:border-nova-accent/40">
              <div className="flex items-start justify-between gap-3">
                <span className="flex items-center gap-2 font-display text-[15px] font-medium text-nova-text group-hover:text-nova-accent">
                  {s.status === "active" && <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-nova-success" />}
                  {s.title}
                </span>
                <span className="shrink-0 font-mono text-[9px] uppercase tracking-[0.14em] text-nova-text-dim/70">{s.status}</span>
              </div>
              {s.summary && <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-nova-text-dim">{s.summary}</p>}
              <div className="mt-1.5 font-mono text-[9px] tracking-tight text-nova-text-dim/60">{s.update_count} updates · {s.event_count} events · updated {formatDate(s.last_updated)}</div>
            </button>
          ))}
          {shown.length > limit && (
            <button onClick={() => setLimit((n) => n + 30)}
              className="w-full rounded-lg border border-nova-border bg-nova-surface/40 py-2 font-mono text-[10px] uppercase tracking-[0.14em] text-nova-text-dim transition-colors hover:border-nova-accent/40 hover:text-nova-text">
              Show {Math.min(30, shown.length - limit)} more · {shown.length - limit} remaining
            </button>
          )}
        </div>
      ) : (
        <SwimLane rows={mapRows} hover={hover} setHover={setHover} onOpen={read} />
      )}
    </div>
  );
}

/** The swimlane thread-map — one lane per thread, dots at each update, a NOW
 *  divider. Bigger/brighter type because the viewBox scales down ~0.7×. */
function SwimLane({
  rows, hover, setHover, onOpen,
}: {
  rows: { s: StorylineSummary; events: StorylineEvent[] }[];
  hover: { e: StorylineEvent; title: string } | null;
  setHover: (h: { e: StorylineEvent; title: string } | null) => void;
  onOpen: (s: StorylineSummary) => void;
}) {
  const W = 1000, ML = 250, MR = 26, MT = 30, rowH = 38;
  const layout = useMemo(() => {
    const allT: number[] = [];
    rows.forEach((t) => {
      const lu = Date.parse(t.s.last_updated); if (!Number.isNaN(lu)) allT.push(lu);
      t.events.forEach((e) => { const v = Date.parse(e.created_at); if (!Number.isNaN(v)) allT.push(v); });
    });
    const now = Date.now(); allT.push(now);
    if (allT.length < 2) return null;
    let tMin = Math.min(...allT), tMax = Math.max(...allT);
    const pad = (tMax - tMin) * 0.05 || 86400000; tMin -= pad; tMax += pad;
    const xOf = (t: number) => ML + ((t - tMin) / (tMax - tMin)) * (W - ML - MR);
    const H = MT * 2 + rows.length * rowH;
    const ticks = [0, 0.33, 0.66, 1].map((p) => ({ x: ML + p * (W - ML - MR), label: new Date(tMin + p * (tMax - tMin)).toLocaleDateString("en-US", { month: "short", day: "numeric" }) }));
    return { xOf, H, nowX: xOf(now), ticks };
  }, [rows]);

  const lead = hover ?? (rows[0]?.events[0] ? { e: rows[0].events[0], title: rows[0].s.title } : null);

  return (
    <>
      {/* min-w + x-scroll: the viewBox scales text down uniformly, so on narrow
          panels the plot scrolls sideways instead of shrinking to unreadable */}
      <div className="overflow-x-auto rounded-lg border border-nova-border bg-nova-bg/50">
        {layout && (
          <svg viewBox={`0 0 ${W} ${layout.H}`} className="w-full min-w-[640px]" style={{ display: "block" }}>
            <line x1={layout.nowX} y1={MT - 8} x2={layout.nowX} y2={layout.H - MT + 8} stroke="#e08a3c" strokeWidth={1.5} strokeDasharray="5 5" opacity={0.75} />
            <text x={layout.nowX} y={16} textAnchor="middle" fontSize={13} fill="#e08a3c" fontFamily="monospace" letterSpacing="2">NOW</text>
            {layout.ticks.map((t, i) => (<text key={i} x={t.x} y={16} textAnchor="middle" fontSize={14} fill="#b0aa9a" fontFamily="monospace">{t.label}</text>))}
            {rows.map((th, i) => {
              const y = MT + i * rowH + rowH / 2;
              let evs = th.events.map((e) => ({ e, x: layout.xOf(Date.parse(e.created_at)) })).filter((p) => !Number.isNaN(p.x));
              if (evs.length === 0) {
                const lu = Date.parse(th.s.last_updated);
                if (!Number.isNaN(lu)) evs = [{ e: { id: -th.s.id, summary: th.s.summary, created_at: th.s.last_updated, is_new: false } as StorylineEvent, x: layout.xOf(lu) }];
              }
              const xs = evs.map((p) => p.x);
              const x0 = xs.length ? Math.min(...xs) : layout.nowX, x1 = xs.length ? Math.max(...xs) : layout.nowX;
              return (
                <g key={th.s.id} style={{ cursor: "pointer" }} onClick={() => onOpen(th.s)}>
                  <text x={ML - 14} y={y + 5} textAnchor="end" fontSize={17} fill="#eae7df" fontFamily="'Space Grotesk Variable', sans-serif">{trunc(th.s.title, 30)}</text>
                  <line x1={x0} y1={y} x2={x1} y2={y} stroke="#5a554a" strokeWidth={2.5} strokeLinecap="round" />
                  {evs.map((p, k) => (
                    <circle key={k} cx={p.x} cy={y} r={hover?.e.id === p.e.id ? 9 : p.e.is_new ? 6 : 5}
                      fill={p.e.is_new ? "#f0a850" : "#b0aa9a"} opacity={p.e.is_new ? 1 : 0.92}
                      onMouseEnter={() => setHover({ e: p.e, title: th.s.title })} onMouseLeave={() => setHover(null)} />
                  ))}
                </g>
              );
            })}
          </svg>
        )}
      </div>
      <div className="mt-4 min-h-[64px] rounded-lg border border-nova-border bg-nova-surface/50 p-3.5 backdrop-blur-md">
        {lead ? (
          <>
            <div className="mb-1 flex flex-wrap items-center gap-2 font-mono text-[10px] uppercase tracking-[0.16em]">
              <span className="text-nova-accent">{trunc(lead.title, 52)}</span>
              <span className="text-nova-text-dim">· {formatDate(lead.e.created_at)}</span>
              {lead.e.is_new && <span className="text-nova-accent/80">· new</span>}
            </div>
            <p className="whitespace-pre-line text-[13px] leading-relaxed text-nova-text">{lead.e.summary}</p>
          </>
        ) : <p className="text-xs text-nova-text-dim">Hover a thread's update to read it, or click a lane to open it.</p>}
      </div>
    </>
  );
}

function trunc(s: string, n: number) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }
