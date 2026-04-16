import { useRef, useCallback, useEffect, useState, useMemo } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { KGGraphData, KGGraphNode, MonitorResultHit, MessageSearchResult } from "../lib/types";
import { Skeleton } from "./ui";
import { Maximize2, ZoomIn, ZoomOut, Network, PanelLeftClose, PanelRightClose, ChevronDown, ChevronRight, Filter, Activity, MessageSquare, Loader2, X } from "lucide-react";
import { formatDate } from "../lib/utils";

/* ── palette ── */
const NODE_COLORS = [
  "#818cf8", "#22d3ee", "#f472b6", "#a78bfa", "#34d399",
  "#fb923c", "#38bdf8", "#fbbf24", "#e879f9", "#4ade80",
];
function colorOf(id: string): string {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = ((h << 5) - h + id.charCodeAt(i)) | 0;
  return NODE_COLORS[Math.abs(h) % NODE_COLORS.length];
}
function pill(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r); ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h); ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r); ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath();
}

export interface EntityInfo {
  monitors: MonitorResultHit[];
  msgs: MessageSearchResult[];
  loading: boolean;
}

interface Props {
  graphData: KGGraphData;
  onNodeClick?: (node: KGGraphNode) => void;
  selectedEntity?: string;
  entityInfo?: EntityInfo;
  loading?: boolean;
  height?: number;
}

export default function KGGraph({ graphData, onNodeClick, selectedEntity, entityInfo, loading, height = 560 }: Props) {
  const fg = useRef<any>(null);
  const box = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(800);
  const [hov, setHov] = useState<string | null>(null);
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(false);
  const [predsOpen, setPredsOpen] = useState(false);
  const [activePred, setActivePred] = useState<string | null>(null);

  const monitors = entityInfo?.monitors ?? [];
  const msgs = entityInfo?.msgs ?? [];
  const infoLoading = entityInfo?.loading ?? false;

  // Open right panel when entity is selected
  useEffect(() => {
    if (selectedEntity) {
      setRightOpen(true);
      setLeftOpen(false);
    }
  }, [selectedEntity]);

  const LEFT_W = leftOpen ? 220 : 0;
  const RIGHT_W = rightOpen && selectedEntity ? 300 : 0;
  const graphW = Math.max(w - LEFT_W - RIGHT_W, 200);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver((e) => setW(Math.floor(e[0].contentRect.width)));
    ro.observe(el);
    setW(el.offsetWidth);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const f = fg.current;
    if (!f) return;
    f.d3Force("charge")?.strength(-180).distanceMax(350);
    f.d3Force("link")?.distance(65);
    f.d3Force("center")?.strength(0.05);
  }, []);

  useEffect(() => {
    const f = fg.current;
    if (!f || graphData.nodes.length === 0) return;
    const n = graphData.nodes.length;
    f.d3Force("charge")?.strength(n > 150 ? -100 : n > 60 ? -140 : -180);
    f.d3Force("link")?.distance(n > 150 ? 45 : n > 60 ? 55 : 65);
    setTimeout(() => { f.centerAt(0, 0, 400); f.zoom(n > 150 ? 0.5 : n > 60 ? 0.7 : 1.0, 400); }, 600);
  }, [graphData]);

  const maxVal = Math.max(...graphData.nodes.map((n) => n.val), 1);

  const topEntities = useMemo(() => [...graphData.nodes].sort((a, b) => b.val - a.val).slice(0, 40), [graphData.nodes]);
  const predicateCounts = useMemo(() => {
    const m: Record<string, number> = {};
    for (const l of graphData.links) m[l.label] = (m[l.label] || 0) + 1;
    return Object.entries(m).sort((a, b) => b[1] - a[1]);
  }, [graphData.links]);

  const selectedFacts = useMemo(() => {
    if (!selectedEntity) return [];
    const sel = selectedEntity.toLowerCase();
    return graphData.links.filter((l) => {
      const s = typeof l.source === "object" ? (l.source as any).id : l.source;
      const t = typeof l.target === "object" ? (l.target as any).id : l.target;
      return s?.toLowerCase() === sel || t?.toLowerCase() === sel;
    }).map((l) => {
      const s = typeof l.source === "object" ? (l.source as any).id : l.source;
      const t = typeof l.target === "object" ? (l.target as any).id : l.target;
      return { source: s, predicate: l.label, target: t, confidence: l.confidence };
    });
  }, [graphData.links, selectedEntity]);

  const displayData = useMemo<KGGraphData>(() => {
    if (!activePred) return graphData;
    const fl = graphData.links.filter((l) => l.label === activePred);
    const ids = new Set<string>();
    for (const l of fl) {
      ids.add(typeof l.source === "object" ? (l.source as any).id : l.source);
      ids.add(typeof l.target === "object" ? (l.target as any).id : l.target);
    }
    return { nodes: graphData.nodes.filter((n) => ids.has(n.id)), links: fl };
  }, [graphData, activePred]);

  const nbrSet = useCallback((id: string | null): Set<string> => {
    if (!id) return new Set();
    const s = new Set<string>();
    for (const l of displayData.links) {
      const a = typeof l.source === "object" ? (l.source as any).id : l.source;
      const b = typeof l.target === "object" ? (l.target as any).id : l.target;
      if (a === id || b === id) { s.add(a); s.add(b); }
    }
    return s;
  }, [displayData.links]);

  const focusId = hov || selectedEntity || null;
  const nbrs = nbrSet(focusId);
  const hasHL = focusId !== null && nbrs.size > 0;

  /* ─── Painters ─── */
  const paintNode = useCallback((node: any, ctx: CanvasRenderingContext2D, gs: number) => {
    if (!isFinite(node.x) || !isFinite(node.y)) return;
    const id: string = node.id ?? "";
    const label: string = node.label || id;
    const isSel = selectedEntity?.toLowerCase() === id.toLowerCase();
    const isHov = hov === id;
    const isAct = isSel || isHov;
    const isNbr = hasHL && nbrs.has(id);
    const isDim = hasHL && !isNbr && !isAct;
    const color = colorOf(id);
    const t = Math.min((node.val || 1) / maxVal, 1);
    const R = 7 + t * 14, r = isAct ? R * 1.2 : R;

    if (isAct) {
      const g = ctx.createRadialGradient(node.x, node.y, r * 0.5, node.x, node.y, r * 2.2);
      g.addColorStop(0, color + "35"); g.addColorStop(1, color + "00");
      ctx.beginPath(); ctx.arc(node.x, node.y, r * 2.2, 0, Math.PI * 2); ctx.fillStyle = g; ctx.fill();
    }
    ctx.beginPath(); ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
    ctx.fillStyle = isDim ? "#222238" : color; ctx.globalAlpha = isDim ? 0.25 : isAct ? 1 : 0.85;
    ctx.fill(); ctx.globalAlpha = 1;
    if (isAct) { ctx.beginPath(); ctx.arc(node.x, node.y, r + 1.5, 0, Math.PI * 2); ctx.strokeStyle = "#ffffffcc"; ctx.lineWidth = 2 / gs; ctx.stroke(); }
    else if (!isDim) { ctx.beginPath(); ctx.arc(node.x, node.y, r, 0, Math.PI * 2); ctx.strokeStyle = color + "44"; ctx.lineWidth = 0.8 / gs; ctx.stroke(); }

    if (isDim && gs < 0.6) return;
    const fs = Math.max(11 / gs, 2.2);
    const cap = gs < 0.4 ? 8 : gs < 0.7 ? 14 : 24;
    const txt = label.length > cap ? label.slice(0, cap - 1) + "\u2026" : label;
    ctx.font = `${isAct ? "700 " : "500 "}${fs}px Inter, system-ui, sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    const ty = node.y + r + 4, tw = ctx.measureText(txt).width, px = 4 / gs, py = 2 / gs;
    pill(ctx, node.x - tw / 2 - px, ty - py, tw + px * 2, fs + py * 2, 4 / gs);
    ctx.fillStyle = isDim ? "rgba(10,10,15,0.2)" : "rgba(6,6,12,0.9)"; ctx.fill();
    ctx.fillStyle = isDim ? "#444" : isAct ? "#fff" : "#bfc1d8"; ctx.fillText(txt, node.x, ty);
  }, [selectedEntity, hov, hasHL, nbrs, maxVal]);

  const paintLink = useCallback((link: any, ctx: CanvasRenderingContext2D, gs: number) => {
    const s = link.source, t = link.target;
    if (!s || !t || !isFinite(s.x) || !isFinite(t.x)) return;
    const sId = typeof s === "object" ? s.id : s, tId = typeof t === "object" ? t.id : t;
    const isAct = hasHL && (sId === focusId || tId === focusId) && nbrs.has(sId) && nbrs.has(tId);
    const isDim = hasHL && !isAct;

    ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(t.x, t.y);
    if (isAct) {
      const g = ctx.createLinearGradient(s.x, s.y, t.x, t.y);
      g.addColorStop(0, colorOf(sId) + "aa"); g.addColorStop(1, colorOf(tId) + "aa");
      ctx.strokeStyle = g; ctx.lineWidth = 2.5 / gs;
    } else if (isDim) { ctx.strokeStyle = "rgba(35,35,55,0.1)"; ctx.lineWidth = 0.3 / gs; }
    else { ctx.strokeStyle = "rgba(120,130,230,0.2)"; ctx.lineWidth = 1 / gs; }
    ctx.stroke();

    const ang = Math.atan2(t.y - s.y, t.x - s.x);
    const tR = 7 + Math.min((t.val || 1) / maxVal, 1) * 14;
    const al = Math.max(5 / gs, 2);
    const ax = t.x - Math.cos(ang) * (tR + 3), ay = t.y - Math.sin(ang) * (tR + 3);
    ctx.beginPath(); ctx.moveTo(ax, ay);
    ctx.lineTo(ax - al * Math.cos(ang - 0.4), ay - al * Math.sin(ang - 0.4));
    ctx.lineTo(ax - al * Math.cos(ang + 0.4), ay - al * Math.sin(ang + 0.4));
    ctx.closePath();
    ctx.fillStyle = isAct ? colorOf(tId) + "bb" : isDim ? "rgba(35,35,55,0.06)" : "rgba(120,130,230,0.15)"; ctx.fill();

    if (link.label && (isAct || gs > 2) && !isDim) {
      const mx = (s.x + t.x) / 2, my = (s.y + t.y) / 2, fs = Math.max(8 / gs, 2);
      const lbl = link.label.replace(/_/g, " ");
      ctx.font = `500 ${fs}px Inter, system-ui, sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      const lw = ctx.measureText(lbl).width, p = 3 / gs;
      pill(ctx, mx - lw / 2 - p, my - fs / 2 - p, lw + p * 2, fs + p * 2, 3 / gs);
      ctx.fillStyle = "rgba(6,6,12,0.94)"; ctx.fill();
      ctx.fillStyle = isAct ? "#fff" : "#9a9cc0"; ctx.fillText(lbl, mx, my);
    }
  }, [hasHL, focusId, nbrs, maxVal]);

  const zoomFit = () => fg.current?.zoomToFit(400, 30);
  const zoomIn = () => { const f = fg.current; if (f) f.zoom(f.zoom() * 1.4, 200); };
  const zoomOut = () => { const f = fg.current; if (f) f.zoom(f.zoom() / 1.4, 200); };
  const focusOn = (entity: string) => onNodeClick?.({ id: entity, label: entity, val: 1 });

  const toggleLeft = () => { setLeftOpen(!leftOpen); if (!leftOpen) setRightOpen(false); };
  const toggleRight = () => { setRightOpen(!rightOpen); if (!rightOpen) setLeftOpen(false); };

  return (
    <div ref={box} className="relative flex rounded-lg border border-nova-border overflow-hidden" style={{ height }}>

      {/* ═══ LEFT: Entity browser ═══ */}
      {leftOpen && (
        <div className="shrink-0 border-r border-nova-border bg-nova-surface/60 backdrop-blur-sm overflow-y-auto animate-slide-in-left" style={{ width: 220 }}>
          <div className="p-3 space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-[10px] font-semibold uppercase tracking-wider text-nova-text-dim">Entities</h4>
              <button onClick={() => setLeftOpen(false)} className="text-nova-text-dim hover:text-nova-text"><X size={12} /></button>
            </div>
            <div className="space-y-0.5">
              {topEntities.map((n) => (
                <button key={n.id} onClick={() => focusOn(n.id)}
                  className={`flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs transition-all hover:bg-nova-accent/10 ${
                    selectedEntity?.toLowerCase() === n.id.toLowerCase() ? "bg-nova-accent/15 text-nova-text" : "text-nova-text-dim hover:text-nova-text"
                  }`}>
                  <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: colorOf(n.id) }} />
                  <span className="truncate flex-1">{n.label}</span>
                  <span className="shrink-0 text-[10px] text-nova-text-dim/60">{n.val}</span>
                </button>
              ))}
            </div>

            <div>
              <button onClick={() => setPredsOpen(!predsOpen)}
                className="flex w-full items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-nova-text-dim mb-1 hover:text-nova-text transition-colors">
                {predsOpen ? <ChevronDown size={10} /> : <ChevronRight size={10} />} Relationships
              </button>
              {predsOpen && (
                <div className="space-y-0.5">
                  <button onClick={() => setActivePred(null)}
                    className={`flex w-full items-center gap-2 rounded-md px-2 py-1 text-left text-[11px] transition-all ${!activePred ? "bg-nova-accent/15 text-nova-accent" : "text-nova-text-dim hover:text-nova-text hover:bg-nova-accent/5"}`}>
                    <Filter size={10} /> All
                  </button>
                  {predicateCounts.map(([pred, cnt]) => (
                    <button key={pred} onClick={() => setActivePred(activePred === pred ? null : pred)}
                      className={`flex w-full items-center justify-between rounded-md px-2 py-1 text-left text-[11px] transition-all ${activePred === pred ? "bg-nova-accent/15 text-nova-accent" : "text-nova-text-dim hover:text-nova-text hover:bg-nova-accent/5"}`}>
                      <span className="truncate">{pred.replace(/_/g, " ")}</span>
                      <span className="shrink-0 text-[10px] text-nova-text-dim/50">{cnt}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ═══ CENTER: Graph ═══ */}
      <div className="relative flex-1 min-w-0" style={{ background: "radial-gradient(ellipse at 50% 40%, #141430 0%, #0d0d1a 45%, #0a0a0f 100%)" }}>
        {loading && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-nova-bg/60 backdrop-blur-sm">
            <div className="flex items-center gap-2 text-sm text-nova-text-dim"><Loader2 size={16} className="animate-spin" /> Loading graph...</div>
          </div>
        )}

        <div className="absolute top-3 left-3 z-10 flex gap-1">
          <button onClick={toggleLeft} className={`rounded-md border bg-nova-surface/80 backdrop-blur-sm p-1.5 transition-all ${leftOpen ? "border-nova-accent/30 text-nova-accent" : "border-nova-border text-nova-text-dim hover:text-nova-text"}`} title="Entity browser">
            <PanelLeftClose size={13} />
          </button>
        </div>

        <div className="absolute top-3 right-3 z-10 flex gap-1">
          {selectedEntity && (
            <button onClick={toggleRight} className={`rounded-md border bg-nova-surface/80 backdrop-blur-sm p-1.5 transition-all ${rightOpen ? "border-nova-accent/30 text-nova-accent" : "border-nova-border text-nova-text-dim hover:text-nova-text"}`} title="Entity info">
              <PanelRightClose size={13} />
            </button>
          )}
          <button onClick={zoomIn} className="rounded-md border border-nova-border bg-nova-surface/80 backdrop-blur-sm p-1.5 text-nova-text-dim hover:text-nova-text transition-all" title="Zoom in"><ZoomIn size={13} /></button>
          <button onClick={zoomOut} className="rounded-md border border-nova-border bg-nova-surface/80 backdrop-blur-sm p-1.5 text-nova-text-dim hover:text-nova-text transition-all" title="Zoom out"><ZoomOut size={13} /></button>
          <button onClick={zoomFit} className="rounded-md border border-nova-border bg-nova-surface/80 backdrop-blur-sm p-1.5 text-nova-text-dim hover:text-nova-text transition-all" title="Fit all"><Maximize2 size={13} /></button>
          <span className="rounded-md border border-nova-border bg-nova-surface/80 backdrop-blur-sm px-2 py-1.5 text-[10px] text-nova-text-dim font-medium">
            {displayData.nodes.length} &middot; {displayData.links.length}
          </span>
        </div>

        {activePred && (
          <div className="absolute bottom-3 left-3 z-10 flex items-center gap-2 rounded-md border border-nova-accent/30 bg-nova-surface/80 backdrop-blur-sm px-2.5 py-1.5">
            <Filter size={10} className="text-nova-accent" />
            <span className="text-[11px] text-nova-accent font-medium">{activePred.replace(/_/g, " ")}</span>
            <button onClick={() => setActivePred(null)} className="text-nova-text-dim hover:text-nova-text text-xs ml-1">&times;</button>
          </div>
        )}

        {!loading && graphData.nodes.length > 0 && (
          <ForceGraph2D
            ref={fg} graphData={displayData} width={graphW} height={height} backgroundColor="rgba(0,0,0,0)"
            nodeCanvasObject={paintNode}
            nodePointerAreaPaint={(node: any, c: string, ctx: CanvasRenderingContext2D) => {
              const t = Math.min((node.val || 1) / maxVal, 1);
              ctx.beginPath(); ctx.arc(node.x, node.y, 9 + t * 16, 0, Math.PI * 2); ctx.fillStyle = c; ctx.fill();
            }}
            linkCanvasObject={paintLink}
            linkPointerAreaPaint={(link: any, c: string, ctx: CanvasRenderingContext2D) => {
              const s = link.source, t = link.target; if (!s || !t || s.x == null) return;
              ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(t.x, t.y); ctx.strokeStyle = c; ctx.lineWidth = 8; ctx.stroke();
            }}
            onNodeClick={(node: any) => onNodeClick?.({ id: node.id, label: node.label || node.id, val: node.val || 1 })}
            onNodeHover={(node: any) => { setHov(node?.id ?? null); if (box.current) box.current.style.cursor = node ? "pointer" : "grab"; }}
            d3AlphaDecay={0.015} d3VelocityDecay={0.25} cooldownTicks={250} warmupTicks={100}
            enableNodeDrag={true} enableZoomInteraction={true} enablePanInteraction={true}
          />
        )}

        {!loading && graphData.nodes.length === 0 && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <Network size={44} strokeWidth={1.5} className="text-nova-accent/40" />
            <p className="text-sm text-nova-text-dim">No graph data</p>
          </div>
        )}
      </div>

      {/* ═══ RIGHT: Entity detail ═══ */}
      {rightOpen && selectedEntity && (
        <div className="shrink-0 border-l border-nova-border bg-nova-surface/60 backdrop-blur-sm overflow-y-auto animate-slide-in-right" style={{ width: 300 }}>
          <div className="p-4 space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-3.5 w-3.5 rounded-full" style={{ backgroundColor: colorOf(selectedEntity), boxShadow: `0 0 8px ${colorOf(selectedEntity)}66` }} />
                <h3 className="text-sm font-bold text-nova-text">{selectedEntity}</h3>
              </div>
              <button onClick={() => setRightOpen(false)} className="text-nova-text-dim hover:text-nova-text"><X size={14} /></button>
            </div>

            {infoLoading && (
              <div className="flex items-center gap-2 py-3 text-xs text-nova-text-dim"><Loader2 size={12} className="animate-spin" /> Searching...</div>
            )}

            {/* Monitor Results — the real intelligence */}
            {monitors.length > 0 && (
              <div>
                <h4 className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-nova-text-dim mb-2">
                  <Activity size={10} /> Monitor Intelligence ({monitors.length})
                </h4>
                <div className="space-y-2">
                  {monitors.map((m) => (
                    <div key={m.id} className="rounded-lg border border-nova-border/60 bg-nova-bg/60 p-3">
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-[10px] font-medium text-nova-accent/80">{m.monitor_name}</span>
                        <span className="text-[9px] text-nova-text-dim/50">{formatDate(m.created_at)}</span>
                      </div>
                      <p className="text-[12px] text-nova-text/90 leading-relaxed whitespace-pre-line">
                        {m.content.slice(0, 400)}{m.content.length > 400 ? "\u2026" : ""}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Conversation mentions */}
            {msgs.length > 0 && (
              <div>
                <h4 className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-nova-text-dim mb-2">
                  <MessageSquare size={10} /> Conversations ({msgs.length})
                </h4>
                <div className="space-y-2">
                  {msgs.map((m, i) => (
                    <div key={i} className="rounded-lg border border-nova-border/60 bg-nova-bg/60 p-3">
                      <p className="text-[12px] text-nova-text/80 leading-relaxed">
                        {m.content.slice(0, 300)}{m.content.length > 300 ? "\u2026" : ""}
                      </p>
                      <p className="text-[9px] text-nova-text-dim/50 mt-1">{m.role} &middot; {m.created_at?.slice(0, 10)}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Connections */}
            {selectedFacts.length > 0 && (
              <div>
                <h4 className="text-[10px] font-semibold uppercase tracking-wider text-nova-text-dim mb-2">
                  Connections ({selectedFacts.length})
                </h4>
                <div className="space-y-0.5">
                  {selectedFacts.map((f, i) => {
                    const isSrc = f.source.toLowerCase() === selectedEntity.toLowerCase();
                    const other = isSrc ? f.target : f.source;
                    return (
                      <button key={i} onClick={() => focusOn(other)}
                        className="flex w-full items-center gap-1.5 rounded px-2 py-1.5 text-left text-[11px] text-nova-text-dim hover:text-nova-text hover:bg-nova-accent/8 transition-all">
                        <span className="text-nova-accent/70 text-[9px] shrink-0 w-16 truncate">{f.predicate.replace(/_/g, " ")}</span>
                        <span className="text-nova-text-dim/40 shrink-0">{isSrc ? "\u2192" : "\u2190"}</span>
                        <span className="truncate font-medium text-nova-text/90">{other}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {!infoLoading && monitors.length === 0 && msgs.length === 0 && selectedFacts.length === 0 && (
              <p className="text-xs text-nova-text-dim/50 py-4 text-center">No information found for this entity.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
