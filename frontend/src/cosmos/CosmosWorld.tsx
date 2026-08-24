import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import { EffectComposer, Bloom, Vignette, ChromaticAberration, Noise } from "@react-three/postprocessing";
import { BlendFunction } from "postprocessing";
import * as THREE from "three";
import { Field, softSprite, type EntityFact } from "../components/KnowledgeCosmos";
import { getKGGraph } from "../lib/api";
import type { KGGraphData, KGGraphNode } from "../lib/types";
import { REGIONS, regionById, type RegionId } from "./regions";

const CA_OFFSET = new THREE.Vector2(0.0006, 0.0009);

/** Flies the camera: a slow orbit over the whole galaxy in overview, a swoop in
 *  to the selected region's waypoint when one is focused. Damped, cinematic. */
function CameraRig({ focus }: { focus: RegionId | null }) {
  const { camera } = useThree();
  const look = useRef(new THREE.Vector3(0, 0, 0));
  const tmpP = useRef(new THREE.Vector3());
  const tmpL = useRef(new THREE.Vector3());

  useFrame((state, dt) => {
    const p = tmpP.current, l = tmpL.current;
    const region = regionById(focus);
    if (!region) {
      // overview — a stable, gently-swaying 3/4 view that frames the galaxy with
      // the waypoint crown above it (full orbit made the composition inconsistent)
      const t = state.clock.elapsedTime;
      p.set(Math.sin(t * 0.07) * 130, 470, 980);
      l.set(0, 60, 0);
      p.x += state.pointer.x * 40;
      p.y += -state.pointer.y * 22;
    } else {
      // fly to just outside the waypoint, looking back through it at the galaxy
      const wp = new THREE.Vector3(...region.pos);
      const radial = wp.clone().setY(0).normalize();
      p.copy(wp).add(radial.multiplyScalar(230)).add(new THREE.Vector3(0, 130, 0));
      l.copy(wp);
      p.x += state.pointer.x * 22;
      p.y += -state.pointer.y * 15;
    }
    const k = Math.min(1, dt * 1.5);
    camera.position.lerp(p, k);
    look.current.lerp(l, k);
    camera.lookAt(look.current);
  });
  return null;
}

/** A single region waypoint: a glowing beacon + a clickable label, floating in
 *  the cosmos. Dims when another region holds focus. */
function Waypoint({
  id, label, tag, color, pos, focus, onSelect, count,
}: {
  id: RegionId; label: string; tag: string; color: string; pos: [number, number, number];
  focus: RegionId | null; onSelect: (id: RegionId) => void; count?: string;
}) {
  const soft = useMemo(softSprite, []);
  const [hover, setHover] = useState(false);
  const beacon = useRef<THREE.Sprite>(null);
  const isFocus = focus === id;
  const dim = focus !== null && !isFocus;

  useFrame((state) => {
    const s = 58 + Math.sin(state.clock.elapsedTime * 1.4 + pos[0]) * 6;
    if (beacon.current) beacon.current.scale.setScalar(hover ? s * 1.25 : s);
  });

  return (
    <group position={pos}>
      <sprite ref={beacon} scale={[58, 58, 1]}>
        <spriteMaterial map={soft} color={color} transparent opacity={dim ? 0.16 : hover ? 0.95 : 0.62} depthWrite={false} blending={THREE.AdditiveBlending} />
      </sprite>
      <sprite scale={[14, 14, 1]}>
        <spriteMaterial map={soft} color={color} transparent opacity={dim ? 0.4 : 1} depthWrite={false} blending={THREE.AdditiveBlending} />
      </sprite>
      {!isFocus && (
        <Html center distanceFactor={1000} zIndexRange={[30, 0]} style={{ pointerEvents: "auto" }}>
          <button
            onClick={() => onSelect(id)}
            onPointerOver={() => setHover(true)}
            onPointerOut={() => setHover(false)}
            className={`group flex select-none items-center gap-2 whitespace-nowrap rounded-full border px-3.5 py-1.5 transition-all ${dim ? "opacity-45" : "opacity-100"}`}
            style={{
              transform: "translateY(36px)",
              borderColor: hover ? color : "rgba(234,231,223,0.2)",
              background: "rgba(8,8,10,0.62)",
              boxShadow: hover ? `0 0 22px ${color}55` : "none",
            }}
          >
            <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: color, boxShadow: `0 0 8px ${color}` }} />
            <span className="font-display text-[15px] font-semibold tracking-[-0.01em]" style={{ color: hover ? color : "#eae7df" }}>{label}</span>
            {count
              ? <span className="font-mono text-[11px] font-medium tabular-nums" style={{ color }}>{count}</span>
              : <span className="font-mono text-[8.5px] uppercase tracking-[0.22em] text-nova-text-dim">{tag}</span>}
          </button>
        </Html>
      )}
    </group>
  );
}

function Scene({ focus, onSelect, data, counts, onPick, selected }: { focus: RegionId | null; onSelect: (id: RegionId) => void; data: KGGraphData | null; counts?: Partial<Record<RegionId, string>>; onPick?: (node: KGGraphNode, facts: EntityFact[]) => void; selected?: string | null }) {
  return (
    <>
      <fog attach="fog" args={["#0b0b0d", 300, 1100]} />
      <Field data={data} onPick={onPick} selected={selected} interactive={focus == null} />
      {REGIONS.map((r) => (
        <Waypoint key={r.id} {...r} focus={focus} onSelect={onSelect} count={counts?.[r.id]} />
      ))}
      <CameraRig focus={focus} />
      <EffectComposer>
        <Bloom intensity={1.05} luminanceThreshold={0.5} luminanceSmoothing={0.28} mipmapBlur radius={0.72} />
        <ChromaticAberration blendFunction={BlendFunction.NORMAL} offset={CA_OFFSET} radialModulation={false} modulationOffset={0} />
        <Vignette offset={0.12} darkness={focus ? 1.15 : 1.02} eskil={false} />
        <Noise premultiply blendFunction={BlendFunction.OVERLAY} opacity={0.025} />
      </EffectComposer>
    </>
  );
}

/** The persistent 3D world. One galaxy, region waypoints, a cinematic camera. */
export default function CosmosWorld({ focus, overview, onSelect, counts, onPick, selected }: { focus: RegionId | null; overview: boolean; onSelect: (id: RegionId) => void; counts?: Partial<Record<RegionId, string>>; onPick?: (node: KGGraphNode, facts: EntityFact[]) => void; selected?: string | null }) {
  const [data, setData] = useState<KGGraphData | null>(null);
  useEffect(() => {
    let alive = true;
    getKGGraph(undefined, 2, 950).then((d) => { if (alive) setData(d); }).catch(() => {});
    return () => { alive = false; };
  }, []);

  return (
    <Canvas
      camera={{ position: [0, 1100, 2100], fov: 55, near: 1, far: 3400 }}
      dpr={[1, 1.5]}
      gl={{ antialias: false, alpha: false, powerPreference: "high-performance", stencil: false }}
      raycaster={{ params: { Points: { threshold: 12 } } } as any}
      // Only the overview is interactive. Whenever a panel is open (any region OR
      // Systems) the whole 3D layer (canvas + waypoint <Html> buttons) goes
      // pointer-events:none, so nothing in the galaxy behind the panel can steal
      // clicks/hover. The panel is then the only interactive surface.
      style={{ position: "absolute", inset: 0, pointerEvents: overview ? "auto" : "none" }}
    >
      <color attach="background" args={["#08080a"]} />
      <Scene focus={focus} onSelect={onSelect} data={data} counts={counts} onPick={onPick} selected={selected} />
    </Canvas>
  );
}
