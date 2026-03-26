import { Search } from "lucide-react";
import { Card, EmptyState, Skeleton } from "../../components/ui";
import { formatDate } from "../../lib/utils";
import type { CuriosityItem } from "../../lib/types";

interface Props {
  items: CuriosityItem[];
  loading: boolean;
}

export default function CuriositySection({ items, loading }: Props) {
  if (loading) {
    return <Skeleton lines={4} />;
  }

  if (items.length === 0) {
    return (
      <EmptyState
        icon={<Search size={40} strokeWidth={1.5} />}
        title="No curiosity items queued."
        description="Nova generates curiosity items when it detects knowledge gaps during conversation."
      />
    );
  }

  return (
    <section className="space-y-2">
      {items.map((item) => (
        <Card key={item.id}>
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium">{item.question}</p>
              <div className="mt-1 flex flex-wrap items-center gap-2">
                <span className="rounded bg-nova-border/40 px-1.5 py-0.5 text-[10px] text-nova-text-dim">
                  {item.source}
                </span>
                <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                  item.status === "completed" || item.status === "researched"
                    ? "bg-nova-success/20 text-nova-success"
                    : item.status === "pending"
                    ? "bg-nova-border text-nova-text-dim"
                    : "bg-nova-warning/20 text-nova-warning"
                }`}>
                  {item.status}
                </span>
                <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                  item.priority >= 3 ? "bg-nova-error/20 text-nova-error"
                    : item.priority >= 2 ? "bg-nova-warning/20 text-nova-warning"
                    : "bg-nova-border text-nova-text-dim"
                }`}>
                  {item.priority >= 3 ? "High" : item.priority >= 2 ? "Med" : "Low"} priority
                </span>
              </div>
            </div>
            <span className="shrink-0 text-xs text-nova-text-dim">
              {formatDate(item.created_at)}
            </span>
          </div>
        </Card>
      ))}
    </section>
  );
}
