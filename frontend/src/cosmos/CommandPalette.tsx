import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { REGIONS, type RegionId } from "./regions";

interface Item { id: string; label: string; sub: string; go: RegionId | "systems" }

const BASE: Item[] = [
  ...REGIONS.map((r) => ({ id: r.id, label: r.label, sub: r.tag, go: r.id as RegionId })),
  { id: "systems", label: "Systems", sub: "settings · monitors · learning", go: "systems" as const },
];

/** ⌘K — jump to any region instantly instead of flying. Portaled so it clears
 *  the glass region panels' stacking context. */
export default function CommandPalette({ open, onClose, onGo }: { open: boolean; onClose: () => void; onGo: (id: RegionId | "systems") => void }) {
  const [q, setQ] = useState("");
  const [idx, setIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const items = useMemo(() => {
    const n = q.trim().toLowerCase();
    return n ? BASE.filter((i) => `${i.label} ${i.sub}`.toLowerCase().includes(n)) : BASE;
  }, [q]);

  useEffect(() => { if (open) { setQ(""); setIdx(0); const t = setTimeout(() => inputRef.current?.focus(), 20); return () => clearTimeout(t); } }, [open]);
  useEffect(() => { setIdx(0); }, [q]);

  if (!open) return null;
  const run = (i: Item) => { onGo(i.go); onClose(); };

  return createPortal(
    <div className="fixed inset-0 z-[60] flex items-start justify-center bg-black/50 pt-[15vh] backdrop-blur-sm animate-fade-in" onClick={onClose}>
      <div className="w-full max-w-lg overflow-hidden rounded-xl border border-nova-border bg-[#0b0b0d]/95 shadow-2xl backdrop-blur-xl animate-scale-in" onClick={(e) => e.stopPropagation()}>
        <input
          ref={inputRef}
          role="combobox"
          aria-expanded="true"
          aria-controls="palette-list"
          aria-activedescendant={items[idx] ? `palette-opt-${items[idx].id}` : undefined}
          aria-label="Fly to a region"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setIdx((i) => Math.min(items.length - 1, i + 1)); }
            else if (e.key === "ArrowUp") { e.preventDefault(); setIdx((i) => Math.max(0, i - 1)); }
            else if (e.key === "Enter" && items[idx]) run(items[idx]);
            else if (e.key === "Escape") onClose();
          }}
          placeholder="Fly to a region…"
          className="w-full border-b border-nova-border bg-transparent px-4 py-3.5 text-sm text-nova-text outline-none placeholder:text-nova-text-dim/60"
        />
        <div id="palette-list" role="listbox" aria-label="Regions" className="max-h-[50vh] overflow-y-auto p-1.5">
          {items.map((i, k) => (
            <button key={i.id} id={`palette-opt-${i.id}`} role="option" aria-selected={k === idx}
              onMouseEnter={() => setIdx(k)} onClick={() => run(i)}
              className={`flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-left transition-colors ${k === idx ? "bg-nova-accent/10" : "hover:bg-nova-surface/40"}`}>
              <span className={`font-display text-sm ${k === idx ? "text-nova-accent" : "text-nova-text"}`}>{i.label}</span>
              <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-nova-text-dim/70">{i.sub}</span>
            </button>
          ))}
          {items.length === 0 && <div className="px-3 py-6 text-center text-sm text-nova-text-dim">No match</div>}
        </div>
      </div>
    </div>,
    document.body,
  );
}
