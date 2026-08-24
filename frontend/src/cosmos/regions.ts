/**
 * The regions of Nova's mind — the facets you fly to inside the cosmos. There is
 * no nav rail and no pages; these are waypoints placed in 3D space around the
 * knowledge galaxy. Selecting one flies the camera to it and opens it in-world.
 */
export type RegionId =
  | "knowledge"
  | "signals"
  | "forecasts"
  | "storylines"
  | "dossiers"
  | "chat";

export interface Region {
  id: RegionId;
  label: string;
  tag: string;
  color: string;
  pos: [number, number, number];
}

const RING = 560; // at the galaxy's edge — bright beacons keep them distinct from the stars
const DEFS: Omit<Region, "pos">[] = [
  { id: "knowledge", label: "Knowledge", tag: "the graph", color: "#f0a850" },
  { id: "signals", label: "Signals", tag: "briefings", color: "#7fb7e8" },
  { id: "forecasts", label: "Forecasts", tag: "predictions", color: "#79d89b" },
  { id: "storylines", label: "Storylines", tag: "threads", color: "#c98fe0" },
  { id: "dossiers", label: "Dossiers", tag: "understanding", color: "#f2c879" },
  { id: "chat", label: "Interrogate", tag: "ask Nova", color: "#e0894a" },
];

// Hand-placed as a top arc (angles chosen so five sit across the upper frame and
// the sixth stays on the RIGHT) — the lower-left is reserved for the World Brief.
// θ: 0=right, 90=front/bottom, 270=back/top. y elevated into a crown.
const ANGLES = [200, 236, 272, 308, 344, 22].map((d) => (d * Math.PI) / 180);
export const REGIONS: Region[] = DEFS.map((d, i) => {
  const a = ANGLES[i];
  return { ...d, pos: [Math.cos(a) * RING, i % 2 ? 196 : 128, Math.sin(a) * RING] };
});

export const REGION_IDS = REGIONS.map((r) => r.id);
export function regionById(id: RegionId | null): Region | undefined {
  return REGIONS.find((r) => r.id === id);
}
