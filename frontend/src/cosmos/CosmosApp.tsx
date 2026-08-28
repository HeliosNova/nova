import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { X, Settings2, Lock } from "lucide-react";
import CosmosWorld from "./CosmosWorld";
import { REGIONS, regionById, type RegionId } from "./regions";
import { useWorldBrief, briefParts, fmtBriefDate } from "../pages/WorldBrief";
import { getMonitors, getRecentMonitorResults, getForecasts, getStorylines, getKGStats } from "../lib/api";
import type { MonitorInfo, MonitorResult, KGGraphNode } from "../lib/types";
import type { EntityFact } from "../components/KnowledgeCosmos";
import { useChatStore } from "../lib/store";
import CommandPalette from "./CommandPalette";
import ForecastsField from "./ForecastsField";
import StorylinesMap from "./StorylinesMap";
import SignalsBriefing from "./SignalsBriefing";

const ChatPanel = lazy(() => import("./ChatPanel"));
const DossiersPanel = lazy(() => import("./DossiersPanel"));
const SystemsPanel = lazy(() => import("./SystemsPanel"));
const KnowledgeGraphView = lazy(() => import("./KnowledgeGraphView"));

type Focus = RegionId | "systems" | null;
const VALID = new Set<string>([...REGIONS.map((r) => r.id), "systems"]);
function hashFocus(): Focus {
  const h = window.location.hash.replace("#", "");
  return VALID.has(h) ? (h as Focus) : null;
}

interface Telemetry { forecasts: number; fcAcc: number | null; storylines: number; facts: number; entities: number }

export default function CosmosApp() {
  const [focus, setFocus] = useState<Focus>(hashFocus);
  const [monitors, setMonitors] = useState<MonitorInfo[]>([]);
  const [results, setResults] = useState<MonitorResult[]>([]);
  const [resultsLoading, setResultsLoading] = useState(true);
  const [tel, setTel] = useState<Telemetry | null>(null);
  const [palette, setPalette] = useState(false);
  const [picked, setPicked] = useState<{ node: KGGraphNode; facts: EntityFact[] } | null>(null);
  const [authLocked, setAuthLocked] = useState(false);
  const brief = useWorldBrief();

  // Backend 401s: every data caller swallows its own error, so without this
  // the cosmos shows confident zeros when the browser simply has no API key.
  useEffect(() => {
    const lock = () => setAuthLocked(true);
    const unlock = () => setAuthLocked(false);
    window.addEventListener("nova:auth-required", lock);
    window.addEventListener("nova:auth-ok", unlock);
    return () => {
      window.removeEventListener("nova:auth-required", lock);
      window.removeEventListener("nova:auth-ok", unlock);
    };
  }, []);

  // hash <-> focus (deep-linkable, back-button friendly)
  useEffect(() => {
    const on = () => setFocus(hashFocus());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const go = (id: Focus) => { window.location.hash = id ?? ""; setFocus(id); if (id) setPicked(null); };

  // Keyboard focus follows the panel: on open, move focus to its close button
  // (the waypoints live in the 3D layer, which goes inert); on close, restore
  // whatever had focus before the flight in.
  const closeRef = useRef<HTMLButtonElement>(null);
  const prevFocus = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (focus) {
      if (!prevFocus.current && document.activeElement instanceof HTMLElement) prevFocus.current = document.activeElement;
      const t = setTimeout(() => closeRef.current?.focus(), 40);
      return () => clearTimeout(t);
    }
    if (prevFocus.current?.isConnected) prevFocus.current.focus();
    prevFocus.current = null;
  }, [focus]);

  // Keyboard nav: Esc flies back to the cosmos; 1–6 fly to a region.
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      const typing = !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); setPalette((p) => !p); return; }
      if (e.key === "Escape") {
        if (palette) setPalette(false);
        // While a chat response streams, let ChatPage's own Esc stop it instead
        // of flying out of the region (both listen on window).
        else if (focus === "chat" && useChatStore.getState().streaming) { /* chat handles stop */ }
        else if (focus) go(null);
        return;
      }
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
      const n = parseInt(e.key, 10);
      if (n >= 1 && n <= REGIONS.length) go(REGIONS[n - 1].id);
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [focus, palette]);

  useEffect(() => {
    getMonitors().then((v) => setMonitors(Array.isArray(v) ? v : [])).catch(() => {});
    getRecentMonitorResults(72).then((v) => setResults(Array.isArray(v) ? v : [])).catch(() => {}).finally(() => setResultsLoading(false));
    Promise.all([getForecasts(1).catch(() => null), getStorylines("active", 1).catch(() => null), getKGStats().catch(() => null)])
      .then(([f, s, k]) => setTel({
        forecasts: f?.stats.open ?? 0,
        fcAcc: f?.stats.accuracy ?? null,
        storylines: s?.stats.active ?? 0,
        facts: k?.current_facts ?? 0,
        entities: k?.unique_entities ?? 0,
      })).catch(() => {});
  }, []);

  const region = regionById(focus === "systems" ? null : focus);
  const isSystems = focus === "systems";
  const panelLabel = region?.label ?? "Systems";
  const panelTag = region?.tag ?? "configuration";
  const panelColor = region?.color ?? "#eae7df";
  const enabled = useMemo(() => monitors.filter((m) => m.enabled).length, [monitors]);
  const date = brief ? fmtBriefDate(brief.updated) : "";

  // live counts shown on the waypoints so the overview reads at a glance
  const compact = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1).replace(/\.0$/, "")}k` : String(n));
  const counts: Partial<Record<RegionId, string>> = {
    knowledge: tel ? compact(tel.facts) : "",
    forecasts: tel ? String(tel.forecasts) : "",
    storylines: tel ? String(tel.storylines) : "",
    signals: monitors.length ? String(enabled) : "",
  };

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-[#08080a] text-nova-text">
      {/* THE WORLD */}
      <CosmosWorld
        focus={focus === "systems" ? null : focus}
        overview={focus === null}
        onSelect={(id) => go(id)}
        counts={counts}
        onPick={(node, facts) => setPicked({ node, facts })}
        selected={picked?.node.id ?? null}
      />

      {/* corner instrument frame */}
      <span className="hud-bracket tl" /><span className="hud-bracket tr" />
      <span className="hud-bracket bl" /><span className="hud-bracket br" />

      {/* auth lock — the backend is refusing this browser, say so plainly */}
      {authLocked && (
        <div className="pointer-events-none absolute inset-x-0 top-24 z-20 flex justify-center">
          <button
            onClick={() => go("systems")}
            className="pointer-events-auto flex items-center gap-2 rounded-md border border-nova-accent/40 bg-nova-surface/70 px-4 py-2 font-mono text-[10px] uppercase tracking-[0.22em] text-nova-accent backdrop-blur-md transition-colors hover:border-nova-accent/70"
          >
            <Lock size={11} />
            backend locked — set the API key in Systems ▸ Settings
          </button>
        </div>
      )}

      {/* HUD — identity + where you are */}
      <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start justify-between p-6 sm:p-8">
        <button onClick={() => go(null)} className="pointer-events-auto flex items-center gap-2.5 text-left">
          <span className="relative flex h-2.5 w-2.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-nova-accent opacity-50" />
            <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-nova-accent shadow-[0_0_10px_rgba(224,138,60,0.55)]" />
          </span>
          <span className="font-display text-lg font-semibold tracking-[-0.02em]">NOVA</span>
          <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-nova-text-dim/80">
            {focus ? <>Cosmos <span className="text-nova-border">▸</span> <span className="text-nova-accent">{panelLabel}</span></> : "Knowledge Cosmos"}
          </span>
        </button>
        <div className="flex items-center gap-4">
          <div className="hidden text-right font-mono text-[10px] uppercase tracking-[0.18em] text-nova-text-dim/80 sm:block">
            <div>{date || "—"}</div>
            <div className="mt-1 text-nova-text-dim/60">Watching <span className="text-nova-text/90">{enabled}</span>/{monitors.length}</div>
          </div>
          <button
            onClick={() => go(isSystems ? null : "systems")}
            className={`pointer-events-auto rounded-md border p-2 backdrop-blur-md transition-colors ${isSystems ? "border-nova-accent/40 text-nova-accent" : "border-nova-border bg-nova-surface/40 text-nova-text-dim hover:text-nova-text"}`}
            title="Systems"
            aria-label="Systems"
          >
            <Settings2 size={16} />
          </button>
        </div>
      </div>

      {/* OVERVIEW — the World Brief thesis is the home of the cosmos */}
      {!focus && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 p-6 sm:p-10">
          <div className="mx-auto max-w-6xl">
            <div className="reveal mb-2.5 font-mono text-[10px] uppercase tracking-[0.24em] text-nova-accent" style={{ animationDelay: "0.4s" }}>State of the World</div>
            {brief ? (
              <p className="reveal max-w-[48ch] font-display text-[19px] font-medium leading-[1.3] tracking-[-0.02em] sm:text-[26px] sm:leading-[1.24]" style={{ animationDelay: "0.55s" }}>
                {briefParts(brief.lead).map((p, i) => p.bold ? <span key={i} className="text-nova-accent">{p.text}</span> : <span key={i}>{p.text}</span>)}
              </p>
            ) : <div className="h-20 max-w-[48ch] animate-pulse rounded bg-nova-surface/40" />}
            {brief && (
              <button
                onClick={() => go("dossiers")}
                className="reveal group pointer-events-auto mt-3 inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.16em] text-nova-accent transition-colors hover:text-nova-accent-hover"
                style={{ animationDelay: "0.7s" }}
              >
                Read Nova's full understanding
                <span className="transition-transform group-hover:translate-x-0.5">→</span>
              </button>
            )}
            <div className="reveal mt-6 flex flex-wrap gap-x-8 gap-y-3 border-t border-nova-border/60 pt-4 font-mono text-nova-text-dim" style={{ animationDelay: "0.8s" }}>
              <Stat label="Knowledge" value={tel ? tel.facts.toLocaleString() : "—"} sub="facts" accent />
              <Stat label="Entities" value={tel ? tel.entities.toLocaleString() : "—"} sub="tracked" />
              <Stat label="Forecasts" value={tel ? String(tel.forecasts) : "—"} sub={tel?.fcAcc != null ? `${Math.round(tel.fcAcc * 100)}% graded` : "open"} />
              <Stat label="Storylines" value={tel ? String(tel.storylines) : "—"} sub="active" />
            </div>
            <div className="reveal mt-4 font-mono text-[10px] uppercase tracking-[0.2em] text-nova-text-dim/75" style={{ animationDelay: "1s" }}>
              Select a region to fly in · press 1–{REGIONS.length} or ⌘K · move to explore
            </div>
          </div>
        </div>
      )}

      {/* FOCUSED — the region opens in-world as a glass console. Knowledge (the
          interactive graph) gets a much wider stage. */}
      {(region || isSystems) && (
        <div role="region" aria-label={panelLabel} className={`absolute right-0 top-0 z-40 flex h-full w-full flex-col border-l border-nova-border bg-[#0b0b0d] shadow-[-24px_0_70px_-18px_rgba(0,0,0,0.92)] animate-slide-in-right ${focus === "knowledge" ? "max-w-[min(97vw,1200px)]" : focus === "chat" ? "max-w-[min(96vw,1060px)]" : "max-w-[min(94vw,760px)]"}`}>
          <div className="flex items-center justify-between border-b border-nova-border px-5 py-3.5">
            <div>
              <div className="font-mono text-[9px] uppercase tracking-[0.24em] text-nova-text-dim/70">{panelTag}</div>
              <h2 className="font-display text-xl font-semibold tracking-[-0.02em]" style={{ color: panelColor }}>{panelLabel}</h2>
            </div>
            <button ref={closeRef} onClick={() => go(null)} className="rounded-md border border-nova-border bg-nova-surface/50 p-1.5 text-nova-text-dim backdrop-blur-md transition-colors hover:border-nova-accent/40 hover:text-nova-text" aria-label="Back to the cosmos" title="Back to the cosmos (Esc)">
              <X size={16} />
            </button>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">
            <Suspense fallback={<div className="p-6"><div className="h-24 animate-pulse rounded bg-nova-surface/40" /></div>}>
              <div className={focus === "chat" || focus === "knowledge" ? "h-full" : focus === "forecasts" || focus === "storylines" || focus === "signals" ? "" : "p-4 sm:p-5"}>
                {focus === "forecasts" && <ForecastsField />}
                {focus === "storylines" && <StorylinesMap />}
                {focus === "signals" && <SignalsBriefing results={results} monitors={monitors} loading={resultsLoading} />}
                {focus === "dossiers" && <DossiersPanel />}
                {focus === "knowledge" && <KnowledgeGraphView />}
                {focus === "chat" && <ChatPanel />}
                {isSystems && <SystemsPanel />}
              </div>
            </Suspense>
          </div>
        </div>
      )}

      {/* Entity card — click a star in the galaxy to inspect it in-world */}
      {picked && (
        <div className="absolute right-4 top-24 z-30 flex max-h-[68vh] w-full max-w-[320px] flex-col rounded-xl border border-nova-border bg-[#0b0b0d] shadow-[0_20px_60px_-15px_rgba(0,0,0,0.85)] animate-scale-in">
          <div className="flex items-start justify-between gap-2 border-b border-nova-border px-4 py-3">
            <div className="min-w-0">
              <div className="font-mono text-[9px] uppercase tracking-[0.2em] text-nova-accent">Entity</div>
              <h3 className="truncate font-display text-lg font-semibold tracking-[-0.01em] text-nova-text">{picked.node.label}</h3>
              <div className="mt-0.5 font-mono text-[10px] tracking-tight text-nova-text-dim">{picked.facts.length} connections traced</div>
            </div>
            <button onClick={() => setPicked(null)} className="shrink-0 rounded-md border border-nova-border p-1 text-nova-text-dim transition-colors hover:text-nova-text" aria-label="Deselect">
              <X size={14} />
            </button>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto p-1.5">
            {picked.facts.slice(0, 80).map((f, i) => (
              <div key={i} className="flex items-baseline gap-2 rounded-md px-2 py-1.5 text-xs">
                <span className="shrink-0 font-mono text-[9px] uppercase tracking-[0.08em] text-nova-accent/80">{f.out ? "→" : "←"} {f.predicate.replace(/_/g, " ")}</span>
                <span className="truncate text-nova-text">{f.other}</span>
              </div>
            ))}
            {picked.facts.length === 0 && <p className="px-2 py-8 text-center text-xs text-nova-text-dim">No connections recorded for this entity.</p>}
          </div>
          <div className="border-t border-nova-border px-4 py-2.5">
            <button onClick={() => go("knowledge")} className="font-mono text-[10px] uppercase tracking-[0.16em] text-nova-accent transition-colors hover:text-nova-accent-hover">
              Open in the graph tool →
            </button>
          </div>
        </div>
      )}

      <CommandPalette open={palette} onClose={() => setPalette(false)} onGo={(id) => go(id)} />
    </div>
  );
}

function Stat({ label, value, sub, accent }: { label: string; value: string; sub?: string; accent?: boolean }) {
  return (
    <div>
      <div className="text-[9px] uppercase tracking-[0.2em] text-nova-text-dim/80">{label}</div>
      <div className={`mt-0.5 text-lg tabular-nums tracking-tight ${accent ? "text-nova-accent" : "text-nova-text"}`}>{value}</div>
      {sub && <div className="text-[9px] tracking-tight text-nova-text-dim/60">{sub}</div>}
    </div>
  );
}
