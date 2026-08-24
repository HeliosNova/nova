import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import { EffectComposer, Bloom, Vignette, ChromaticAberration, Noise } from "@react-three/postprocessing";
import { BlendFunction } from "postprocessing";
import * as THREE from "three";
import { getKGGraph } from "../lib/api";
import type { KGGraphData } from "../lib/types";

/**
 * KnowledgeCosmos — Nova's knowledge graph as a living galaxy you arrive inside.
 *
 * The immersive hero (2026-08-21). Nova is a mind that watches the world and
 * accumulates understanding; its knowledge graph IS that mind's interior. So we
 * render the real graph as a cosmos: every fact is a star, communities are
 * constellations that clump in their own region of space and wear their domain
 * name, connectivity sets a star's size and heat, and a faint neural web of the
 * strongest links threads between them.
 *
 * The one hard idea (award research: pick one, execute cleanly) is that the
 * structure is REAL — not decorative particles. The craft around it: a cinematic
 * arrival (camera dollies in, stars ignite), living turbulence + twinkle in the
 * vertex shader, cursor parallax, and a restrained film grade (bloom + faint
 * chromatic aberration + vignette + grain). One draw call for the whole field.
 */

const COMM_COLORS = [
  "#f0a850", "#e08a3c", "#f2c879", "#c96f3a",
  "#7fb7e8", "#5e8fd6",
  "#c98fe0", "#8f7fe0",
  "#6fd6b0", "#e0d06f",
  "#e87f9a", "#d0d0e0",
];

// ChromaticAberration wants a Vector2 (static → build once).
const CA_OFFSET = new THREE.Vector2(0.0006, 0.0009);

function rnd(i: number, seed = 0) {
  const x = Math.sin(i * 127.1 + seed * 311.7) * 43758.5453;
  return x - Math.floor(x);
}

interface Label { pos: [number, number, number]; text: string; members: number }
interface CosmosField {
  positions: Float32Array;
  colors: Float32Array;
  sizes: Float32Array;
  linkPositions: Float32Array | null;
  labels: Label[];
  nebulae: Nebula[];
  count: number;
  // interactivity: the real KG nodes sit at indices 0..nodeCount-1 (ambient after)
  nodeCount: number;
  nodeList: import("../lib/types").KGGraphNode[];
  idIndex: Map<string, number>;
}

// A dense field of faint far stars behind the real graph — gives the void depth
// and makes it read as deep space, not a sparse diagram. Appended into the same
// buffer so it stays a single draw call.
// A dense, centre-weighted galactic disc of stars — additive overlap builds a
// natural luminous haze, so the frame reads as deep space, not scattered dots.
const AMBIENT = 16000;
function pushAmbient(positions: Float32Array, colors: Float32Array, sizes: Float32Array, off: number) {
  const warm = new THREE.Color("#d8c8a8");
  const cool = new THREE.Color("#9fb6d8");
  const ember = new THREE.Color("#e8a86a");
  for (let i = 0; i < AMBIENT; i++) {
    const j = off + i;
    // centre-weighted radius (pow<1 pulls toward core) + spiral swirl
    const rr = Math.pow(rnd(i, 31), 0.62);
    const r = 40 + rr * 560;
    const arm = Math.sin(rr * 6.28 * 1.5) * 0.6; // faint spiral bias
    const theta = rnd(i, 32) * Math.PI * 2 + arm;
    positions[j * 3] = r * Math.cos(theta) + (rnd(i, 39) - 0.5) * 60;
    positions[j * 3 + 1] = (rnd(i, 33) - 0.5) * 2 * (36 + r * 0.14); // thin, flared disc
    positions[j * 3 + 2] = r * Math.sin(theta) + (rnd(i, 40) - 0.5) * 60;
    // brighter toward the core; a sparse few burn hot enough to bloom
    const coreGlow = 1 - rr; // 1 at centre → 0 at rim
    const hot = rnd(i, 34) > 0.9;
    const bright = hot ? 1.1 + rnd(i, 35) * 1.0 : 0.4 + rnd(i, 36) * 0.5 + coreGlow * 0.5;
    const tint = coreGlow > 0.55 ? ember : rnd(i, 38) > 0.72 ? cool : warm;
    const c = tint.clone().multiplyScalar(bright);
    colors[j * 3] = c.r; colors[j * 3 + 1] = c.g; colors[j * 3 + 2] = c.b;
    sizes[j] = 1.3 + rnd(i, 37) * 2.6 + coreGlow * 1.5;
  }
}

// Soft radial sprite — the gas of the nebulae + galactic core (dodges the GPU
// gl_PointSize cap that makes giant points unreliable).
let _soft: THREE.CanvasTexture | null = null;
export function softSprite(): THREE.CanvasTexture {
  if (_soft) return _soft;
  const s = 160;
  const cv = document.createElement("canvas");
  cv.width = cv.height = s;
  const ctx = cv.getContext("2d")!;
  const g = ctx.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.2, "rgba(255,255,255,0.55)");
  g.addColorStop(0.5, "rgba(255,255,255,0.16)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, s, s);
  _soft = new THREE.CanvasTexture(cv);
  return _soft;
}

interface Nebula { pos: [number, number, number]; color: string; size: number; opacity: number }

function buildField(data: KGGraphData | null): CosmosField {
  const nodes = data?.nodes ?? [];
  const links = data?.links ?? [];
  const legend = new Map<number, string>((data?.communities ?? []).map((c) => [c.id, c.title]));

  if (nodes.length === 0) {
    const positions = new Float32Array(AMBIENT * 3);
    const colors = new Float32Array(AMBIENT * 3);
    const sizes = new Float32Array(AMBIENT);
    pushAmbient(positions, colors, sizes, 0);
    return { positions, colors, sizes, linkPositions: null, labels: [], nebulae: [{ pos: [0, 0, 0], color: "#e8a860", size: 640, opacity: 0.08 }], count: AMBIENT, nodeCount: 0, nodeList: [], idIndex: new Map() };
  }

  const commIds = Array.from(new Set(nodes.map((n) => n.community ?? -1)));
  const centres = new Map<number, THREE.Vector3>();
  const colorOf = new Map<number, THREE.Color>();
  const commOf = new Map<number, number>();
  const memberCount = new Map<number, number>();
  nodes.forEach((n) => {
    const cid = n.community ?? -1;
    memberCount.set(cid, (memberCount.get(cid) ?? 0) + 1);
  });
  commIds.forEach((cid, idx) => {
    const golden = idx * 2.399963;
    const rad = cid === -1 ? 300 : 50 + Math.sqrt(idx + 1) * 52;
    const y = (rnd(idx, 7) - 0.5) * 150;
    centres.set(cid, new THREE.Vector3(Math.cos(golden) * rad, y, Math.sin(golden) * rad));
    colorOf.set(cid, new THREE.Color(cid === -1 ? "#6a5c40" : COMM_COLORS[idx % COMM_COLORS.length]));
    commOf.set(cid, idx);
  });

  const N = nodes.length;
  const total = N + AMBIENT;
  const positions = new Float32Array(total * 3);
  const colors = new Float32Array(total * 3);
  const sizes = new Float32Array(total);
  const idIndex = new Map<string, number>();
  const commByIdx = new Int32Array(N);
  const maxVal = Math.max(1, ...nodes.map((n) => n.val || 1));

  nodes.forEach((n, i) => {
    idIndex.set(n.id, i);
    const cid = n.community ?? -1;
    commByIdx[i] = cid;
    const centre = centres.get(cid)!;
    const spread = cid === -1 ? 170 : 24 + Math.sqrt(n.val || 1) * 3.2;
    const g = () => rnd(i, 10) + rnd(i, 11) + rnd(i, 12) - 1.5;
    positions[i * 3] = centre.x + (rnd(i, 20) - 0.5) * spread * 2;
    positions[i * 3 + 1] = centre.y + g() * spread * 0.75;
    positions[i * 3 + 2] = centre.z + (rnd(i, 22) - 0.5) * spread * 2;

    // brightness floor so every fact is a visible star; hubs run hot for bloom
    const heat = (n.val || 1) / maxVal;
    const base = colorOf.get(cid)!.clone().multiplyScalar(1.15 + heat * 2.9);
    colors[i * 3] = base.r; colors[i * 3 + 1] = base.g; colors[i * 3 + 2] = base.b;
    sizes[i] = 3.6 + Math.sqrt(n.val || 1) * 3.3;
  });

  pushAmbient(positions, colors, sizes, N);

  // Neural web: only INTRA-constellation links (short, structural) — long
  // cross-volume chords are what turn a graph into a hairball.
  const intra = links.filter((l) => {
    const a = idIndex.get(l.source), b = idIndex.get(l.target);
    return a != null && b != null && commByIdx[a] === commByIdx[b] && commByIdx[a] !== -1;
  });
  const strong = intra.sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0)).slice(0, 260);
  const linkPositions = new Float32Array(strong.length * 6);
  strong.forEach((l, k) => {
    const a = idIndex.get(l.source)!, b = idIndex.get(l.target)!;
    for (let j = 0; j < 3; j++) {
      linkPositions[k * 6 + j] = positions[a * 3 + j];
      linkPositions[k * 6 + 3 + j] = positions[b * 3 + j];
    }
  });

  const labels: Label[] = commIds
    .filter((cid) => cid !== -1 && legend.has(cid))
    .map((cid) => {
      const c = centres.get(cid)!;
      return { pos: [c.x, c.y + 22, c.z] as [number, number, number], text: legend.get(cid)!, members: memberCount.get(cid) ?? 0 };
    })
    .sort((a, b) => b.members - a.members)
    .slice(0, 6);

  // Nebula gas: a warm galactic core + a coloured cloud over each big
  // constellation, so the structure sits in luminous gas, not black void.
  const nebulae: Nebula[] = [{ pos: [0, 0, 0], color: "#e8a860", size: 680, opacity: 0.075 }];
  commIds
    .filter((cid) => cid !== -1)
    .map((cid) => ({ cid, m: memberCount.get(cid) ?? 0 }))
    .sort((a, b) => b.m - a.m)
    .slice(0, 9)
    .forEach(({ cid, m }) => {
      const c = centres.get(cid)!;
      nebulae.push({
        pos: [c.x, c.y, c.z],
        color: `#${colorOf.get(cid)!.getHexString()}`,
        size: Math.min(430, 150 + Math.sqrt(m) * 46),
        opacity: 0.13,
      });
    });

  return { positions, colors, sizes, linkPositions: strong.length ? linkPositions : null, labels, nebulae, count: total, nodeCount: N, nodeList: nodes, idIndex };
}

const VERT = /* glsl */ `
  attribute float size;
  varying vec3 vColor;
  uniform float uTime;
  uniform float uOpacity;
  void main() {
    vColor = color;
    vec3 p = position;
    // gentle turbulence — the galaxy breathes rather than sitting rigid
    p.x += sin(uTime * 0.18 + position.z * 0.010) * 2.4;
    p.y += cos(uTime * 0.15 + position.x * 0.012) * 2.0;
    p.z += sin(uTime * 0.17 + position.y * 0.011) * 2.4;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    float twinkle = 0.78 + 0.22 * sin(uTime * 1.6 + position.x * 0.5 + position.y * 0.3);
    gl_PointSize = size * (330.0 / -mv.z) * twinkle * uOpacity;
    gl_Position = projectionMatrix * mv;
  }
`;
const FRAG = /* glsl */ `
  varying vec3 vColor;
  uniform float uOpacity;
  void main() {
    float d = length(gl_PointCoord - vec2(0.5));
    if (d > 0.5) discard;
    float core = smoothstep(0.5, 0.0, d);
    float glow = pow(core, 1.6);
    gl_FragColor = vec4(vColor * glow, glow * uOpacity);
  }
`;

export interface EntityFact { predicate: string; other: string; out: boolean }
export function Field({ data, onPick, selected, interactive = false }: {
  data: KGGraphData | null;
  onPick?: (node: import("../lib/types").KGGraphNode, facts: EntityFact[]) => void;
  selected?: string | null;
  interactive?: boolean;
}) {
  const group = useRef<THREE.Group>(null);
  const field = useMemo(() => buildField(data), [data]);
  const [hover, setHover] = useState<number | null>(null);

  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(field.positions, 3));
    g.setAttribute("color", new THREE.BufferAttribute(field.colors, 3));
    g.setAttribute("size", new THREE.BufferAttribute(field.sizes, 1));
    return g;
  }, [field]);

  const mat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: { uTime: { value: 0 }, uOpacity: { value: 0 } },
        vertexShader: VERT,
        fragmentShader: FRAG,
        vertexColors: true,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    []
  );

  const linkGeom = useMemo(() => {
    if (!field.linkPositions) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(field.linkPositions, 3));
    return g;
  }, [field]);

  useFrame((state, dt) => {
    const t = state.clock.elapsedTime;
    mat.uniforms.uTime.value = t;
    // fade the whole field in over the first ~2.4s (arrival)
    mat.uniforms.uOpacity.value = Math.min(1, t / 2.4);
    if (group.current) {
      group.current.rotation.y += dt * 0.017;
      // cursor parallax — felt, not seen: small damped tilt toward the pointer
      const px = state.pointer.x, py = state.pointer.y;
      group.current.rotation.x += (py * 0.08 - group.current.rotation.x) * 0.03;
      group.current.position.x += (px * 14 - group.current.position.x) * 0.03;
      group.current.position.y += (-py * 8 - group.current.position.y) * 0.03;
    }
  });

  const soft = useMemo(softSprite, []);
  const posOf = (i: number): [number, number, number] => [field.positions[i * 3], field.positions[i * 3 + 1], field.positions[i * 3 + 2]];

  // selection highlight: the picked star + edges to its neighbours (from the
  // FULL link set, not just the intra-cluster visual web)
  const highlight = useMemo(() => {
    if (!selected || !data) return null;
    const si = field.idIndex.get(selected);
    if (si == null || si >= field.nodeCount) return null;
    const P = field.positions;
    const sp: [number, number, number] = [P[si * 3], P[si * 3 + 1], P[si * 3 + 2]];
    const edges: number[] = [];
    const marks: [number, number, number][] = [];
    const seen = new Set<string>();
    for (const l of data.links) {
      const other = l.source === selected ? l.target : l.target === selected ? l.source : null;
      if (!other || seen.has(other)) continue;
      const oi = field.idIndex.get(other);
      if (oi == null || oi >= field.nodeCount) continue;
      seen.add(other);
      const op: [number, number, number] = [P[oi * 3], P[oi * 3 + 1], P[oi * 3 + 2]];
      edges.push(sp[0], sp[1], sp[2], op[0], op[1], op[2]);
      marks.push(op);
      if (marks.length >= 40) break;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(edges), 3));
    return { sp, marks, geom: g };
  }, [selected, data, field]);

  const factsOf = (id: string): EntityFact[] => {
    const out: EntityFact[] = [];
    if (!data) return out;
    for (const l of data.links) {
      if (l.source === id) out.push({ predicate: l.label, other: l.target, out: true });
      else if (l.target === id) out.push({ predicate: l.label, other: l.source, out: false });
    }
    return out;
  };

  useEffect(() => () => { geom.dispose(); mat.dispose(); linkGeom?.dispose(); }, [geom, mat, linkGeom]);
  useEffect(() => () => highlight?.geom.dispose(), [highlight]);

  const pick = interactive
    ? {
        onPointerMove: (e: any) => { e.stopPropagation(); const i = e.index ?? -1; if (i >= 0 && i < field.nodeCount) { if (i !== hover) setHover(i); document.body.style.cursor = "pointer"; } },
        onPointerOut: () => { setHover(null); document.body.style.cursor = "auto"; },
        onClick: (e: any) => { e.stopPropagation(); const i = e.index ?? -1; if (i >= 0 && i < field.nodeCount) { const n = field.nodeList[i]; onPick?.(n, factsOf(n.id)); } },
      }
    : {};

  return (
    <group ref={group}>
      {field.nebulae.map((n, i) => (
        <sprite key={`neb${i}`} position={n.pos} scale={[n.size, n.size, 1]}>
          <spriteMaterial map={soft} color={n.color} transparent opacity={n.opacity} depthWrite={false} blending={THREE.AdditiveBlending} />
        </sprite>
      ))}
      <points geometry={geom} material={mat} {...pick} />
      {linkGeom && (
        <lineSegments geometry={linkGeom}>
          <lineBasicMaterial color="#e0a860" transparent opacity={0.085} blending={THREE.AdditiveBlending} depthWrite={false} />
        </lineSegments>
      )}
      {field.labels.map((l, i) => (
        <Html key={i} position={l.pos} center style={{ pointerEvents: "none" }} zIndexRange={[6, 0]}>
          <div className="whitespace-nowrap font-mono text-[7.5px] uppercase tracking-[0.18em] text-nova-text-dim/30">
            {l.text}
          </div>
        </Html>
      ))}
      {interactive && hover != null && field.nodeList[hover] && (
        <Html position={posOf(hover)} center style={{ pointerEvents: "none" }} zIndexRange={[50, 0]}>
          <div className="-translate-y-4 whitespace-nowrap rounded-full border border-nova-border bg-nova-bg/85 px-2.5 py-0.5 font-mono text-[10px] text-nova-text backdrop-blur-sm">
            {field.nodeList[hover].label}
          </div>
        </Html>
      )}
      {highlight && (
        <group>
          <lineSegments geometry={highlight.geom}>
            <lineBasicMaterial color="#f2c879" transparent opacity={0.5} blending={THREE.AdditiveBlending} depthWrite={false} />
          </lineSegments>
          <sprite position={highlight.sp} scale={[40, 40, 1]}>
            <spriteMaterial map={soft} color="#ffffff" transparent opacity={0.95} depthWrite={false} blending={THREE.AdditiveBlending} />
          </sprite>
          {highlight.marks.map((m, i) => (
            <sprite key={i} position={m} scale={[15, 15, 1]}>
              <spriteMaterial map={soft} color="#f2c879" transparent opacity={0.85} depthWrite={false} blending={THREE.AdditiveBlending} />
            </sprite>
          ))}
        </group>
      )}
    </group>
  );
}

export type CosmosMode = "hero" | "ambient";

/** Cinematic camera rig. Dollies in from far on load, eases between modes when
 *  the app changes page (hero = close + interactive on the Bulletin; ambient =
 *  a calm high wide shot behind every other page), and hands off to orbit once
 *  settled in hero mode. */
function Rig({ mode }: { mode: CosmosMode }) {
  const { camera } = useThree();
  const [live, setLive] = useState(false);
  const ty = mode === "hero" ? 92 : 140;
  const tz = mode === "hero" ? 300 : 470;

  useEffect(() => { setLive(false); }, [mode]);

  useFrame((_, dt) => {
    if (live) return; // once settled, OrbitControls (hero) or rest (ambient) own the camera
    const k = Math.min(1, dt * 1.7);
    camera.position.x += (0 - camera.position.x) * k;
    camera.position.y += (ty - camera.position.y) * k;
    camera.position.z += (tz - camera.position.z) * k;
    camera.lookAt(0, 0, 0);
    if (Math.hypot(camera.position.y - ty, camera.position.z - tz) < 6) setLive(true);
  });

  if (mode === "hero" && live) {
    return (
      <OrbitControls
        enablePan={false}
        enableZoom
        enableRotate
        enableDamping
        dampingFactor={0.06}
        minDistance={150}
        maxDistance={620}
        rotateSpeed={0.5}
      />
    );
  }
  return null;
}

export default function KnowledgeCosmos({ mode = "hero" }: { mode?: CosmosMode }) {
  const [data, setData] = useState<KGGraphData | null>(null);

  useEffect(() => {
    let alive = true;
    getKGGraph(undefined, 2, 950).then((d) => { if (alive) setData(d); }).catch(() => {});
    return () => { alive = false; };
  }, []);

  return (
    <Canvas
      camera={{ position: [0, 210, 900], fov: 55, near: 1, far: 2600 }}
      dpr={[1, 1.5]}
      gl={{ antialias: false, alpha: true, powerPreference: "high-performance", stencil: false }}
      style={{ background: "transparent" }}
    >
      <fog attach="fog" args={["#0b0b0d", 260, 940]} />
      <Field data={data} />
      <Rig mode={mode} />
      <EffectComposer>
        {/* selective HDR bloom — only the hot, well-connected stars ignite */}
        <Bloom intensity={mode === "hero" ? 1.1 : 0.98} luminanceThreshold={0.5} luminanceSmoothing={0.28} mipmapBlur radius={0.72} />
        <ChromaticAberration blendFunction={BlendFunction.NORMAL} offset={CA_OFFSET} radialModulation={false} modulationOffset={0} />
        <Vignette offset={0.12} darkness={mode === "hero" ? 1.02 : 1.18} eskil={false} />
        <Noise premultiply blendFunction={BlendFunction.OVERLAY} opacity={0.025} />
      </EffectComposer>
    </Canvas>
  );
}
