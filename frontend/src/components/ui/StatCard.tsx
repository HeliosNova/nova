import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface Props {
  label: string;
  value: string | number;
  sub?: string;
  icon?: ReactNode;
  className?: string;
}

export default function StatCard({ label, value, sub, icon, className }: Props) {
  return (
    <div className={cn("rounded-lg border border-nova-border bg-nova-surface/70 px-4 py-3 backdrop-blur-md transition-colors hover:border-nova-border-bright", className)}>
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-medium uppercase tracking-wider text-nova-text-dim">{label}</span>
        {icon && <span className="text-nova-text-dim/70">{icon}</span>}
      </div>
      {/* tabular-nums + tight tracking = the precise, "engineered" KPI numeral */}
      <div className="mt-1.5 text-2xl font-semibold tabular-nums tracking-tight text-nova-text">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-nova-text-dim tabular-nums">{sub}</div>}
    </div>
  );
}
