import { useState, useEffect, useCallback } from "react";
import { Bot, Play, RefreshCw, Filter } from "lucide-react";
import { toast } from "sonner";
import {
  getDaemonStatus,
  getDaemonLog,
  getDaemonEvents,
  triggerDream,
} from "../../lib/api";
import type { DaemonStatus, DaemonLogEntry, EventQueueItem } from "../../lib/types";
import {
  Button,
  Card,
  StatCard,
  Skeleton,
  EmptyState,
  FormSelect,
  ResponsiveTable,
} from "../../components/ui";
import type { Column } from "../../components/ui/ResponsiveTable";
import { formatDate } from "../../lib/utils";

const logColumns: Column<DaemonLogEntry>[] = [
  {
    label: "Category",
    accessor: (e) => (
      <span className="rounded bg-nova-border/40 px-1.5 py-0.5 text-[10px] font-medium text-nova-text-dim">
        {e.category}
      </span>
    ),
  },
  {
    label: "Content",
    accessor: (e) => <span className="text-nova-text-dim">{e.content.slice(0, 120)}</span>,
    className: "max-w-[300px] truncate",
  },
  {
    label: "Source",
    accessor: (e) => <span className="text-xs text-nova-text-dim">{e.source}</span>,
    hideOnMobile: true,
  },
  {
    label: "Time",
    accessor: (e) => <span className="text-xs text-nova-text-dim">{formatDate(e.created_at)}</span>,
  },
];

const eventColumns: Column<EventQueueItem>[] = [
  {
    label: "Type",
    accessor: (e) => <span className="font-medium text-xs">{e.event_type}</span>,
  },
  {
    label: "Priority",
    accessor: (e) => (
      <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
        e.priority >= 0.8 ? "bg-nova-error/20 text-nova-error"
          : e.priority >= 0.5 ? "bg-nova-warning/20 text-nova-warning"
          : "bg-nova-border text-nova-text-dim"
      }`}>
        {e.priority.toFixed(1)}
      </span>
    ),
    className: "text-center",
  },
  {
    label: "Status",
    accessor: (e) => (
      <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
        e.status === "completed" ? "bg-nova-success/20 text-nova-success"
          : e.status === "pending" ? "bg-nova-border text-nova-text-dim"
          : "bg-nova-warning/20 text-nova-warning"
      }`}>
        {e.status}
      </span>
    ),
    className: "text-center",
  },
  {
    label: "Time",
    accessor: (e) => <span className="text-xs text-nova-text-dim">{formatDate(e.created_at)}</span>,
  },
];

export default function DaemonSection() {
  const [status, setStatus] = useState<DaemonStatus | null>(null);
  const [logEntries, setLogEntries] = useState<DaemonLogEntry[]>([]);
  const [events, setEvents] = useState<EventQueueItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [dreaming, setDreaming] = useState(false);
  const [logCategory, setLogCategory] = useState("");
  const [logVisible, setLogVisible] = useState(20);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [s, l, e] = await Promise.all([
        getDaemonStatus().catch(() => null),
        getDaemonLog(24, logCategory || undefined, 100).catch(() => []),
        getDaemonEvents("pending", 50).catch(() => []),
      ]);
      setStatus(s);
      setLogEntries(Array.isArray(l) ? l : []);
      setEvents(Array.isArray(e) ? e : []);
    } finally {
      setLoading(false);
    }
  }, [logCategory]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleDream = async () => {
    setDreaming(true);
    try {
      const result = await triggerDream(false);
      if (result.status === "completed") {
        toast.success("Dream consolidation completed");
      } else {
        toast.info(result.reason || "Dream skipped");
      }
      refresh();
    } catch (err) {
      toast.error(`Dream failed: ${(err as Error).message}`);
    } finally {
      setDreaming(false);
    }
  };

  const logCategories = [...new Set(logEntries.map((e) => e.category))].sort();

  if (loading) {
    return <Skeleton lines={6} />;
  }

  return (
    <section className="space-y-6 animate-fade-in">
      {/* Status cards */}
      {status && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard
            label="Idle Minutes"
            value={status.idle_minutes !== null ? String(Math.round(status.idle_minutes)) : "\u2014"}
          />
          <StatCard
            label="Last Dream"
            value={status.last_dream_at ? formatDate(status.last_dream_at) : "Never"}
          />
          <StatCard label="Pending Events" value={status.pending_events} />
          <StatCard label="Log Entries (24h)" value={status.log_entries_24h} />
        </div>
      )}

      {/* Dream trigger */}
      <div className="flex items-center gap-3">
        <Button
          onClick={handleDream}
          loading={dreaming}
          icon={<Play size={14} />}
          variant="secondary"
        >
          Trigger Dream Consolidation
        </Button>
        <Button onClick={refresh} variant="ghost" size="sm" icon={<RefreshCw size={14} />}>
          Refresh
        </Button>
      </div>

      {/* Event queue */}
      <div>
        <h3 className="mb-2 text-xs font-medium text-nova-text-dim">
          Pending Events ({events.length})
        </h3>
        {events.length === 0 ? (
          <p className="text-sm text-nova-text-dim">No pending events.</p>
        ) : (
          <ResponsiveTable<EventQueueItem>
            columns={eventColumns}
            data={events}
            keyFn={(e) => e.id}
          />
        )}
      </div>

      {/* Daemon log */}
      <div>
        <div className="mb-2 flex items-center gap-3">
          <h3 className="text-xs font-medium text-nova-text-dim">
            Daemon Log ({logEntries.length})
          </h3>
          {logCategories.length > 1 && (
            <FormSelect
              value={logCategory}
              onChange={(e) => { setLogCategory(e.target.value); setLogVisible(20); }}
              placeholder="All categories"
              options={logCategories.map((c) => ({ value: c, label: c }))}
            />
          )}
        </div>
        {logEntries.length === 0 ? (
          <EmptyState
            icon={<Bot size={40} strokeWidth={1.5} />}
            title="No daemon log entries."
            description="Log entries appear when the daemon processes events, runs monitors, or performs maintenance."
          />
        ) : (
          <>
            <ResponsiveTable<DaemonLogEntry>
              columns={logColumns}
              data={logEntries.slice(0, logVisible)}
              keyFn={(e) => e.id}
            />
            {logEntries.length > logVisible && (
              <div className="mt-4 flex justify-center">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setLogVisible((v) => v + 20)}
                >
                  Load More ({logEntries.length - logVisible} remaining)
                </Button>
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}
