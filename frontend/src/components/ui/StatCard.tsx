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
    <div className={cn("rounded-lg border border-nova-border bg-nova-surface px-4 py-3 transition-colors hover:bg-nova-surface/80", className)}>
      <div className="flex items-center justify-between">
        <span className="text-xs text-nova-text-dim">{label}</span>
        {icon && <span className="text-nova-text-dim">{icon}</span>}
      </div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-nova-text-dim">{sub}</div>}
    </div>
  );
}
