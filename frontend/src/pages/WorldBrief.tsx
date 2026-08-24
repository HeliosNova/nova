import { useEffect, useState } from "react";
import { getDossiers, getDossier } from "../lib/api";

export interface WorldBriefData {
  lead: string;
  updated: string;
  revs: number;
}

/** Nova's "State of the World" — its synthesized worldview right now — as data.
 *  Shared by the immersive hero and any inline lead. Pulls the meta dossier's
 *  'Current understanding' and returns the opening 1–2 sentences (the thesis),
 *  with the dossier's own **throughline** markers preserved for emphasis. */
export function useWorldBrief(): WorldBriefData | null {
  const [data, setData] = useState<WorldBriefData | null>(null);
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const metas = await getDossiers("meta");
        const sotw = metas.find((d) => /state of the world/i.test(d.title)) ?? metas[0];
        if (!sotw) return;
        const full = await getDossier(sotw.id);
        const m = (full.body || "").match(/##\s*Current understanding\s*\n([\s\S]*?)(?=\n##\s|\Z)/i);
        const understanding = (m ? m[1] : full.body || "").trim();
        // Accumulate WHOLE sentences up to a comfortable length — never cut mid-word.
        const sentences = understanding.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter(Boolean);
        let lead = "";
        for (const s of sentences) {
          const next = lead ? `${lead} ${s}` : s;
          if (lead && next.length > 300) break;
          lead = next;
          if (lead.length >= 190) break;
        }
        if (alive) setData({ lead, updated: full.updated_at, revs: full.revision_count ?? 0 });
      } catch { /* the brief is a bonus; fail silent */ }
    })();
    return () => { alive = false; };
  }, []);
  return data;
}

/** Split **bold** throughline spans out of the lead for accent rendering. */
export function briefParts(lead: string): { text: string; bold: boolean }[] {
  return lead.split(/(\*\*[^*]+\*\*)/g).filter(Boolean).map((p) =>
    p.startsWith("**") && p.endsWith("**")
      ? { text: p.slice(2, -2), bold: true }
      : { text: p, bold: false }
  );
}

export function fmtBriefDate(iso: string): string {
  return new Date(iso)
    .toLocaleDateString("en-US", { day: "2-digit", month: "short", year: "numeric" })
    .toUpperCase();
}
