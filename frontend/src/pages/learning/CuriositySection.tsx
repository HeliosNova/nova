import { useState } from "react";
import { Search } from "lucide-react";
import { Card, EmptyState, Skeleton } from "../../components/ui";
import { formatDate } from "../../lib/utils";
import type { CuriosityItem } from "../../lib/types";

interface Props {
  items: CuriosityItem[];
  loading: boolean;
}

const RESOLVED = new Set(["resolved", "researched", "completed"]);

export default function CuriositySection({ items, loading }: Props) {
  const [filter, setFilter] = useState<"all" | "pending" | "resolved">("all");

  if (loading) {
    return <Skeleton lines={4} />;
  }

  if (items.length === 0) {
    return (
      <EmptyState
        icon={<Search size={40} strokeWidth={1.5} />}
        title="No curiosity items queued."
        description="Nova generates curiosity items when it detects knowledge gaps — from conversations, dossier open-questions, and unresolved tensions."
      />
    );
  }

  const filtered =
    filter === "all" ? items
    : filter === "pending" ? items.filter(i => i.status === "pending")
    : items.filter(i => RESOLVED.has(i.status));
  const pendingCount = items.filter(i => i.status === "pending").length;
  const resolvedCount = items.filter(i => RESOLVED.has(i.status)).length;

  // urgency is 0–1: >=0.7 High, >=0.5 Med, else Low.
  const urg = (u: number) => u >= 0.7 ? { label: "High", cls: "bg-nova-error/20 text-nova-error" }
    : u >= 0.5 ? { label: "Med", cls: "bg-nova-warning/20 text-nova-warning" }
    : { label: "Low", cls: "bg-nova-border text-nova-text-dim" };

  return (
    <section className="space-y-2">
      <div className="mb-3 flex gap-1.5">
        {([["all", `All (${items.length})`], ["pending", `Pending (${pendingCount})`], ["resolved", `Done (${resolvedCount})`]] as const).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setFilter(key)}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
              filter === key
                ? "bg-nova-accent/15 text-nova-accent border border-nova-accent/30"
                : "text-nova-text-dim hover:text-nova-text hover:bg-nova-border/40 border border-transparent"
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      {filtered.map((item) => {
        const u = urg(item.urgency ?? 0);
        const resolved = RESOLVED.has(item.status);
        return (
          <Card key={item.id}>
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-nova-text leading-snug">{item.topic || "(untitled curiosity)"}</p>
                <div className="mt-1.5 flex flex-wrap items-center gap-2">
                  <span className="rounded bg-nova-border/40 px-1.5 py-0.5 text-[10px] text-nova-text-dim">
                    {item.source?.replace(/_/g, " ")}
                  </span>
                  <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                    resolved ? "bg-nova-success/20 text-nova-success"
                    : item.status === "pending" ? "bg-nova-border text-nova-text-dim"
                    : "bg-nova-warning/20 text-nova-warning"
                  }`}>
                    {item.status}
                  </span>
                  <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${u.cls}`}>{u.label} urgency</span>
                </div>
                {resolved && item.resolution && (
                  <p className="mt-2 border-l-2 border-nova-success/40 pl-2.5 text-[12px] leading-relaxed text-nova-text-dim/85">
                    {item.resolution.length > 320 ? item.resolution.slice(0, 320) + "…" : item.resolution}
                  </p>
                )}
              </div>
              <span className="shrink-0 text-xs text-nova-text-dim">
                {formatDate(item.created_at)}
              </span>
            </div>
          </Card>
        );
      })}
    </section>
  );
}
