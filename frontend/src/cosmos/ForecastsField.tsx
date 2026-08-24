import { useEffect, useMemo, useState } from "react";
import { LayoutGrid, List, Check, X, Clock } from "lucide-react";
import { getForecasts } from "../lib/api";
import type { Forecast, ForecastData } from "../lib/types";

const COLOR = { open: "#e0894a", hit: "#79d89b", miss: "#e8776a" } as const;
const kindOf = (f: Forecast): "open" | "hit" | "miss" => (f.status === "hit" ? "hit" : f.status === "miss" ? "miss" : "open");
const OPEN_SHOWN = 30;

interface Plotted { f: Forecast; x: number; y: number; t: number; kind: "open" | "hit" | "miss" }

const parseDate = (iso?: string | null): Date | null => {
  if (!iso) return null;
  const d = new Date(iso.includes("T") ? iso : iso.replace(" ", "T") + "Z");
  return Number.isNaN(d.getTime()) ? null : d;
};
// forecasts resolve in the FUTURE — formatDate() (which only does relative-past)
// rendered every open one as "just now". These give a real due/graded date.
const dueLabel = (iso?: string | null): string => {
  const d = parseDate(iso);
  if (!d) return "no resolution date";
  const days = Math.round((d.getTime() - Date.now()) / 86_400_000);
  if (days < 0) return `overdue ${-days}d`;
  if (days === 0) return "resolves today";
  if (days === 1) return "resolves tomorrow";
  if (days < 21) return `resolves in ${days} days`;
  if (days < 60) return `resolves in ${Math.round(days / 7)} weeks`;
  return `resolves ${d.toLocaleDateString("en-US", { month: "short", day: "numeric" })}`;
};
const gradedLabel = (iso?: string | null): string => {
  const d = parseDate(iso);
  return d ? `resolved ${d.toLocaleDateString("en-US", { month: "short", day: "numeric" })}` : "resolved";
};

/** Forecasts region: Nova's self-grading predictions. A Text view (default) —
 *  scorecard, then Open predictions (soonest first) and the Hits / Misses track
 *  record, each a clear card with a confidence bar — and a confidence × time
 *  Field, toggled like the storylines map. */
export default function ForecastsField() {
  const [data, setData] = useState<ForecastData | null>(null);
  const [loading, setLoading] = useState(true);
  const [hover, setHover] = useState<Forecast | null>(null);
  const [view, setView] = useState<"text" | "field">("text");

  useEffect(() => { getForecasts(200).then(setData).catch(() => setData(null)).finally(() => setLoading(false)); }, []);

  const W = 1000, H = 440, ML = 46, MR = 20, MT = 26, MB = 40;
  const plot = useMemo(() => {
    if (!data) return null;
    const items = [...data.open, ...data.resolved];
    const ts = (f: Forecast) => parseDate(f.resolved_at || f.resolves_at)?.getTime() ?? null;
    const now = Date.now();
    const times = items.map(ts).filter((v): v is number => v != null).concat(now);
    if (times.length < 2) return null;
    let tMin = Math.min(...times), tMax = Math.max(...times);
    const pad = (tMax - tMin) * 0.06 || 86_400_000;
    tMin -= pad; tMax += pad;
    const xOf = (t: number) => ML + ((t - tMin) / (tMax - tMin)) * (W - ML - MR);
    const yOf = (c: number) => MT + (1 - c) * (H - MT - MB);
    const points: Plotted[] = [];
    for (const f of items) {
      const t = ts(f); if (t == null) continue;
      points.push({ f, x: xOf(t), y: yOf(f.confidence), t, kind: kindOf(f) });
    }
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((p) => { const t = tMin + p * (tMax - tMin); return { x: ML + p * (W - ML - MR), label: new Date(t).toLocaleDateString("en-US", { month: "short", day: "numeric" }) }; });
    return { points, nowX: xOf(now), ticks };
  }, [data]);

  // three plain sections: open (soonest first), then the hit / miss record
  const groups = useMemo(() => {
    if (!data) return { open: [] as Forecast[], hits: [] as Forecast[], misses: [] as Forecast[] };
    const now = Date.now();
    // Open order: soonest UPCOMING first, then overdue (most recent first), then
    // undated — so the top of the list is what Nova is about to be graded on.
    const openOrder = (a: Forecast, b: Forecast) => {
      const ta = parseDate(a.resolves_at)?.getTime(), tb = parseDate(b.resolves_at)?.getTime();
      if (ta == null) return tb == null ? 0 : 1;
      if (tb == null) return -1;
      const fa = ta >= now, fb = tb >= now;
      if (fa !== fb) return fa ? -1 : 1;
      return fa ? ta - tb : tb - ta;
    };
    const recent = (a: Forecast, b: Forecast) => (parseDate(b.resolved_at)?.getTime() ?? 0) - (parseDate(a.resolved_at)?.getTime() ?? 0);
    return {
      open: [...data.open].sort(openOrder),
      hits: data.resolved.filter((f) => f.status === "hit").sort(recent),
      misses: data.resolved.filter((f) => f.status === "miss").sort(recent),
    };
  }, [data]);

  if (loading) return <div className="p-4 sm:p-5"><div className="h-64 animate-pulse rounded-lg bg-nova-surface/40" /></div>;
  if (!data) return <div className="p-8 text-center text-sm text-nova-text-dim">No forecasts yet.</div>;
  const { stats } = data;
  const openShown = groups.open.slice(0, OPEN_SHOWN);
  const moreOpen = Math.max(0, stats.open - openShown.length);

  return (
    <div className="p-4 sm:p-5">
      {/* scorecard + view toggle */}
      <div className="mb-5 flex flex-wrap items-center justify-between gap-x-8 gap-y-3 border-b border-nova-border/60 pb-4">
        <div className="flex flex-wrap gap-x-8 gap-y-3 font-mono">
          <Read label="Open" value={String(stats.open)} tone="#e0894a" />
          <Read label="Hits" value={String(stats.hit)} tone="#79d89b" />
          <Read label="Misses" value={String(stats.miss)} tone="#e8776a" />
          <Read label="Accuracy" value={stats.accuracy != null ? `${Math.round(stats.accuracy * 100)}%` : "—"} sub={stats.graded ? `${stats.hit}/${stats.graded} graded` : "no graded"} />
        </div>
        <div className="flex gap-1">
          {([["text", List, "Text"], ["field", LayoutGrid, "Field"]] as const).map(([v, Icon, lbl]) => (
            <button key={v} onClick={() => setView(v)}
              className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-[0.12em] transition-colors ${view === v ? "border-nova-accent/40 bg-nova-accent/10 text-nova-accent" : "border-nova-border bg-nova-surface/40 text-nova-text-dim hover:text-nova-text"}`}>
              <Icon size={12} /> {lbl}
            </button>
          ))}
        </div>
      </div>

      {view === "text" ? (
        <div>
          <Section label="Open predictions" tone="#e0894a" count={stats.open}>
            {openShown.map((f) => <ForecastCard key={f.id} f={f} />)}
            {moreOpen > 0 && (
              <p className="pt-1 text-center font-mono text-[10px] uppercase tracking-[0.14em] text-nova-text-dim/60">
                + {moreOpen} more open, resolving later
              </p>
            )}
          </Section>
          {groups.hits.length > 0 && (
            <Section label="Hits" tone="#79d89b" count={groups.hits.length}>
              {groups.hits.map((f) => <ForecastCard key={f.id} f={f} />)}
            </Section>
          )}
          {groups.misses.length > 0 && (
            <Section label="Misses" tone="#e8776a" count={groups.misses.length}>
              {groups.misses.map((f) => <ForecastCard key={f.id} f={f} />)}
            </Section>
          )}
          {openShown.length === 0 && groups.hits.length === 0 && groups.misses.length === 0 && (
            <div className="py-12 text-center text-sm text-nova-text-dim">No forecasts on record.</div>
          )}
        </div>
      ) : !plot ? (
        <div className="py-12 text-center text-sm text-nova-text-dim">Not enough dated forecasts to plot yet.</div>
      ) : (
        <>
          {/* min-w + x-scroll so narrow panels scroll the plot instead of
              shrinking its labels to unreadable (viewBox scales uniformly) */}
          <div className="relative overflow-x-auto rounded-lg border border-nova-border bg-nova-bg/40">
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full min-w-[560px]" style={{ display: "block" }}>
              {[0, 0.25, 0.5, 0.75, 1].map((c) => {
                const y = MT + (1 - c) * (H - MT - MB);
                return (
                  <g key={c}>
                    <line x1={ML} y1={y} x2={W - MR} y2={y} stroke="#242320" strokeWidth={1} />
                    <text x={ML - 8} y={y + 3} textAnchor="end" fontSize={15} fill="#b3ae9e" fontFamily="monospace">{Math.round(c * 100)}%</text>
                  </g>
                );
              })}
              {plot.ticks.map((t, i) => (
                <text key={i} x={t.x} y={H - MB + 20} textAnchor="middle" fontSize={15} fill="#b3ae9e" fontFamily="monospace">{t.label}</text>
              ))}
              <line x1={plot.nowX} y1={MT} x2={plot.nowX} y2={H - MB} stroke="#e08a3c" strokeWidth={1} strokeDasharray="4 4" opacity={0.7} />
              <text x={plot.nowX} y={MT - 8} textAnchor="middle" fontSize={13} fill="#e08a3c" fontFamily="monospace" letterSpacing="2">NOW</text>
              {plot.points.map((p, i) => {
                const on = hover?.id === p.f.id;
                return (
                  <g key={i} onMouseEnter={() => setHover(p.f)} onMouseLeave={() => setHover(null)} style={{ cursor: "pointer" }}>
                    <circle cx={p.x} cy={p.y} r={on ? 11 : 6} fill={COLOR[p.kind]} opacity={on ? 0.9 : 0.7} />
                    {on && <circle cx={p.x} cy={p.y} r={16} fill="none" stroke={COLOR[p.kind]} strokeWidth={1} opacity={0.6} />}
                  </g>
                );
              })}
            </svg>
            <div className="pointer-events-none absolute right-3 top-2 flex gap-3 font-mono text-[9px] uppercase tracking-[0.14em]">
              {(["open", "hit", "miss"] as const).map((k) => (
                <span key={k} className="flex items-center gap-1 text-nova-text-dim"><span className="h-2 w-2 rounded-full" style={{ background: COLOR[k] }} />{k}</span>
              ))}
            </div>
          </div>
          <div className="mt-4 min-h-[68px] rounded-lg border border-nova-border bg-nova-surface/50 p-3.5">
            {(() => {
              const f = hover ?? data.open[0] ?? data.resolved[0];
              if (!f) return null;
              const k = kindOf(f);
              return (
                <>
                  <div className="mb-1 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.16em]">
                    <span style={{ color: COLOR[k] }}>{f.status}</span>
                    <span className="text-nova-text-dim">· {Math.round(f.confidence * 100)}% confidence</span>
                    <span className="text-nova-text-dim/70">· {f.status === "open" ? dueLabel(f.resolves_at) : gradedLabel(f.resolved_at)}</span>
                  </div>
                  <p className="text-[13px] leading-relaxed text-nova-text">{f.claim}</p>
                  {f.resolution && <p className="mt-1 text-[11px] italic leading-relaxed text-nova-text-dim/80">{f.resolution}</p>}
                </>
              );
            })()}
          </div>
        </>
      )}
    </div>
  );
}

/** A dossier-style section: a font-display heading with a rule + count. */
function Section({ label, tone, count, children }: { label: string; tone: string; count: number; children: React.ReactNode }) {
  return (
    <section className="mb-6 last:mb-0">
      <div className="mb-2.5 flex items-baseline justify-between gap-3 border-b border-nova-border/50 pb-2">
        <h4 className="font-display text-[17px] font-semibold tracking-[-0.01em]" style={{ color: tone }}>{label}</h4>
        <span className="shrink-0 font-mono text-[10px] uppercase tracking-[0.16em] text-nova-text-dim/60">{count}</span>
      </div>
      <div className="space-y-2">{children}</div>
    </section>
  );
}

/** One forecast: a status badge + real due/graded date on top, the claim, then a
 *  confidence bar so "how sure is Nova" reads at a glance. */
function ForecastCard({ f }: { f: Forecast }) {
  const k = kindOf(f);
  const tone = COLOR[k];
  const conf = Math.round(f.confidence * 100);
  const Icon = k === "hit" ? Check : k === "miss" ? X : Clock;
  const badge = k === "hit" ? "Hit" : k === "miss" ? "Miss" : "Open";
  const when = f.status === "open" ? dueLabel(f.resolves_at) : gradedLabel(f.resolved_at);
  return (
    <div className="rounded-lg border border-nova-border bg-nova-surface/50 p-3.5">
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[9px] font-semibold uppercase tracking-[0.12em]"
          style={{ color: tone, borderColor: `${tone}55`, background: `${tone}18` }}>
          <Icon size={10} strokeWidth={2.5} /> {badge}
        </span>
        <span className="shrink-0 font-mono text-[10px] tracking-tight text-nova-text-dim">{when}</span>
      </div>
      <p className="text-[13.5px] leading-relaxed text-nova-text">{f.claim}</p>
      <div className="mt-2.5 flex items-center gap-2.5">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-nova-border/50">
          <div className="h-full rounded-full" style={{ width: `${conf}%`, background: tone }} />
        </div>
        <span className="shrink-0 font-mono text-[10px] tabular-nums text-nova-text-dim">{conf}% sure</span>
      </div>
      {f.resolution && <p className="mt-2.5 border-l-2 pl-2.5 text-[11.5px] italic leading-relaxed text-nova-text-dim" style={{ borderColor: `${tone}66` }}>{f.resolution}</p>}
    </div>
  );
}

function Read({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: string }) {
  return (
    <div>
      <div className="text-[9px] uppercase tracking-[0.2em] text-nova-text-dim/80">{label}</div>
      <div className="mt-0.5 text-lg tabular-nums tracking-tight" style={{ color: tone ?? "#eae7df" }}>{value}</div>
      {sub && <div className="text-[9px] tracking-tight text-nova-text-dim/60">{sub}</div>}
    </div>
  );
}
