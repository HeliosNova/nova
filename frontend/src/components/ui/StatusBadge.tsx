import { useEffect } from "react";
import { useSettingsStore } from "@/lib/store";
import { getHealth } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  className?: string;
}

export default function StatusBadge({ className }: Props) {
  const health = useSettingsStore((s) => s.health);
  const setHealth = useSettingsStore((s) => s.setHealth);

  useEffect(() => {
    const check = () => {
      getHealth().then(setHealth).catch(() => setHealth(null as never));
    };
    check();
    let interval: ReturnType<typeof setInterval>;
    const start = () => { interval = setInterval(check, 30_000); };
    const stop = () => clearInterval(interval);
    const onVisibility = () => { document.hidden ? stop() : start(); };
    start();
    document.addEventListener("visibilitychange", onVisibility);
    return () => { stop(); document.removeEventListener("visibilitychange", onVisibility); };
  }, [setHealth]);

  const statusText =
    health?.status === "ok"
      ? "Connected"
      : health?.status === "degraded"
        ? "Degraded"
        : "Disconnected";

  const color =
    health?.status === "ok"
      ? "bg-nova-success"
      : health?.status === "degraded"
        ? "bg-nova-warning"
        : "bg-nova-error";

  return (
    <span
      className={cn("inline-flex items-center gap-1.5", className)}
      title={health ? `${health.status} — ${health.model || "no model"}` : "Disconnected"}
    >
      <span className={cn("inline-block h-2.5 w-2.5 rounded-full", color)} />
      <span className="text-xs text-nova-text-dim hidden sm:inline">{statusText}</span>
      <span className="sr-only">{statusText}</span>
    </span>
  );
}
