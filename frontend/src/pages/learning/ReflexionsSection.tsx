import { useState } from "react";
import { GraduationCap } from "lucide-react";
import { Card, EmptyState, Skeleton } from "../../components/ui";
import { formatDate } from "../../lib/utils";
import type { ReflexionInfo } from "../../lib/types";

interface Props {
  reflexions: ReflexionInfo[];
  loading: boolean;
}

type Filter = "all" | "success" | "failure";

export default function ReflexionsSection({ reflexions, loading }: Props) {
  const [filter, setFilter] = useState<Filter>("all");

  if (loading) {
    return <Skeleton lines={4} />;
  }

  if (reflexions.length === 0) {
    return (
      <EmptyState
        icon={<GraduationCap size={40} strokeWidth={1.5} />}
        title="No reflexions recorded yet."
        description="Reflexions are created automatically when Nova assesses its own response quality — both successes and failures."
      />
    );
  }

  const filtered = filter === "all" ? reflexions : reflexions.filter(r => r.outcome === filter);
  const successCount = reflexions.filter(r => r.outcome === "success").length;
  const failureCount = reflexions.filter(r => r.outcome === "failure").length;

  return (
    <section>
      <div className="mb-3 flex gap-1.5">
        {([["all", `All (${reflexions.length})`], ["success", `Successes (${successCount})`], ["failure", `Failures (${failureCount})`]] as const).map(([key, label]) => (
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
      <div className="space-y-3">
      {filtered.map((r) => (
        <Card key={r.id}>
          <div className="mb-1 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span
                className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                  r.outcome === "failure"
                    ? "bg-nova-error/20 text-nova-error"
                    : "bg-nova-success/20 text-nova-success"
                }`}
              >
                {r.outcome}
              </span>
              <span className="text-xs text-nova-text-dim">
                Quality: {Math.round(r.quality_score * 100)}%
              </span>
            </div>
            <span className="text-xs text-nova-text-dim">{formatDate(r.created_at)}</span>
          </div>
          <p className="mb-1 text-sm font-medium">{r.task_summary}</p>
          <p className="text-sm text-nova-text-dim">{r.reflection}</p>
          {r.tools_used.length > 0 && (
            <div className="mt-1 flex gap-1">
              {(typeof r.tools_used === "string" ? (r.tools_used as string).split(",") : r.tools_used)
                .filter(Boolean)
                .map((t: string, i: number) => (
                  <span
                    key={i}
                    className="rounded bg-nova-border px-1.5 py-0.5 text-[10px] text-nova-text-dim"
                  >
                    {t.trim()}
                  </span>
                ))}
            </div>
          )}
        </Card>
      ))}
      </div>
    </section>
  );
}
