import { useEffect, useMemo, useState } from "react";
import { Zap, Search } from "lucide-react";
import { getActions } from "../lib/api";
import { formatDate } from "../lib/utils";
import type { ActionInfo } from "../lib/types";
import {
  PageHeader,
  FormSelect,
  FormInput,
  EmptyState,
  Skeleton,
  StatCard,
  Modal,
  ErrorBanner,
  ResponsiveTable,
} from "../components/ui";
import type { Column } from "../components/ui/ResponsiveTable";

// ── Helpers ──

const HOUR_OPTIONS = [
  { value: "1", label: "Last hour" },
  { value: "6", label: "Last 6 hours" },
  { value: "24", label: "Last 24 hours" },
  { value: "72", label: "Last 3 days" },
  { value: "168", label: "Last week" },
];

function hourLabel(hours: number): string {
  return HOUR_OPTIONS.find((o) => o.value === String(hours))?.label ?? `Last ${hours}h`;
}

function truncate(text: string | null, max: number): string {
  if (!text) return "\u2014";
  return text.length > max ? text.slice(0, max) + "\u2026" : text;
}

function computeTopActionType(actions: ActionInfo[]): string | null {
  if (actions.length === 0) return null;
  const counts: Record<string, number> = {};
  for (const a of actions) {
    counts[a.action_type] = (counts[a.action_type] || 0) + 1;
  }
  let top = "";
  let topCount = 0;
  for (const [type, count] of Object.entries(counts)) {
    if (count > topCount) {
      top = type;
      topCount = count;
    }
  }
  return top || null;
}

// ── Columns ──

const columns: Column<ActionInfo>[] = [
  {
    label: "Time",
    accessor: (a) => (
      <span className="whitespace-nowrap text-nova-text-dim">{formatDate(a.created_at)}</span>
    ),
  },
  {
    label: "Type",
    accessor: (a) => <span className="font-medium">{a.action_type}</span>,
  },
  {
    label: "Status",
    accessor: (a) => (
      <span
        className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
          a.success
            ? "bg-nova-success/20 text-nova-success"
            : "bg-nova-error/20 text-nova-error"
        }`}
      >
        {a.success ? "success" : "error"}
      </span>
    ),
  },
  {
    label: "Params",
    accessor: (a) => (
      <span className="text-nova-text-dim">{truncate(a.params, 60)}</span>
    ),
    className: "max-w-[200px] truncate",
    hideOnMobile: true,
  },
  {
    label: "Result",
    accessor: (a) => (
      <span className="text-nova-text-dim">{truncate(a.result, 80)}</span>
    ),
    className: "max-w-[250px] truncate",
    hideOnMobile: true,
  },
];

// ── Page ──

export default function ActionsPage() {
  const [actions, setActions] = useState<ActionInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filterType, setFilterType] = useState("");
  const [hours, setHours] = useState(24);
  const [searchText, setSearchText] = useState("");
  const [selectedAction, setSelectedAction] = useState<ActionInfo | null>(null);

  const refresh = () => {
    setLoading(true);
    setError(null);
    getActions(filterType || undefined, hours)
      .then((v) => setActions(Array.isArray(v) ? v : []))
      .catch((e) => {
        setActions([]);
        setError(e instanceof Error ? e.message : "Failed to load actions");
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    refresh();
  }, [filterType, hours]);

  // Derived data
  const actionTypes = useMemo(
    () => [...new Set(actions.map((a) => a.action_type))].sort(),
    [actions],
  );

  const filteredActions = useMemo(() => {
    if (!searchText.trim()) return actions;
    const q = searchText.toLowerCase();
    return actions.filter(
      (a) =>
        a.action_type.toLowerCase().includes(q) ||
        (a.params && a.params.toLowerCase().includes(q)) ||
        (a.result && a.result.toLowerCase().includes(q)),
    );
  }, [actions, searchText]);

  const successRate = useMemo(() => {
    if (actions.length === 0) return 0;
    const ok = actions.filter((a) => a.success).length;
    return Math.round((ok / actions.length) * 100);
  }, [actions]);

  const topActionType = useMemo(() => computeTopActionType(actions), [actions]);

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl w-full px-4 md:px-6 py-6 animate-fade-in">
        <PageHeader icon={<Zap size={22} />} title="Action Audit Log" />
        <p className="mb-4 -mt-2 text-xs text-nova-text-dim">
          System actions logged when Nova uses tools, sends alerts, runs monitors, or processes background tasks.
        </p>

        {/* Error banner */}
        {error && (
          <ErrorBanner
            message={error}
            onRetry={refresh}
            onDismiss={() => setError(null)}
          />
        )}

        {/* Stats bar */}
        {!loading && actions.length > 0 && (
          <div className="mb-6 grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatCard label="Total Actions" value={actions.length} />
            <StatCard label="Success Rate" value={`${successRate}%`} />
            <StatCard label="Top Action" value={topActionType || "\u2014"} />
            <StatCard label="Time Range" value={hourLabel(hours)} />
          </div>
        )}

        {/* Filters + Search */}
        <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center">
          <FormInput
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            placeholder="Search actions..."
            icon={<Search size={14} />}
            className="flex-1"
          />
          <div className="flex items-center gap-3">
            <FormSelect
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              placeholder="All types"
              options={actionTypes.map((t) => ({ value: t, label: t }))}
            />
            <FormSelect
              value={String(hours)}
              onChange={(e) => setHours(Number(e.target.value))}
              options={HOUR_OPTIONS}
            />
          </div>
        </div>

        {/* Actions table */}
        {loading ? (
          <Skeleton lines={6} />
        ) : filteredActions.length === 0 ? (
          <EmptyState
            icon={<Zap size={40} strokeWidth={1.5} />}
            title={
              searchText.trim()
                ? "No actions match your search."
                : "No actions in this time range."
            }
            description={
              searchText.trim()
                ? "Try a different search term or clear the filter."
                : "Actions are logged when Nova uses tools, runs monitors, or processes background tasks. Try a longer time range."
            }
          />
        ) : (
          <>
            <div className="mb-2 text-xs text-nova-text-dim">
              {filteredActions.length === actions.length
                ? `${actions.length} action${actions.length !== 1 ? "s" : ""}`
                : `${filteredActions.length} of ${actions.length} actions`}
            </div>
            <ResponsiveTable<ActionInfo>
              columns={columns}
              data={filteredActions}
              keyFn={(a) => a.id}
              onRowClick={(a) => setSelectedAction(a)}
            />
          </>
        )}

        {/* Detail modal */}
        <Modal
          open={selectedAction !== null}
          onClose={() => setSelectedAction(null)}
          title="Action Detail"
          size="lg"
        >
          {selectedAction && (
            <div className="space-y-4">
              {/* Summary grid */}
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-xs text-nova-text-dim">Type</span>
                  <p className="font-medium">{selectedAction.action_type}</p>
                </div>
                <div>
                  <span className="text-xs text-nova-text-dim">Status</span>
                  <p>
                    <span
                      className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                        selectedAction.success
                          ? "bg-nova-success/20 text-nova-success"
                          : "bg-nova-error/20 text-nova-error"
                      }`}
                    >
                      {selectedAction.success ? "success" : "error"}
                    </span>
                  </p>
                </div>
                <div className="col-span-2">
                  <span className="text-xs text-nova-text-dim">Time</span>
                  <p className="font-medium">{formatDate(selectedAction.created_at)}</p>
                </div>
              </div>

              {/* Params block */}
              <div>
                <span className="text-xs font-medium text-nova-text-dim">Params</span>
                <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-all rounded border border-nova-border bg-nova-bg p-3 text-xs text-nova-text-dim">
                  {selectedAction.params || "\u2014"}
                </pre>
              </div>

              {/* Result block */}
              <div>
                <span className="text-xs font-medium text-nova-text-dim">Result</span>
                <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-all rounded border border-nova-border bg-nova-bg p-3 text-xs text-nova-text-dim">
                  {selectedAction.result || "\u2014"}
                </pre>
              </div>
            </div>
          )}
        </Modal>
      </div>
    </div>
  );
}
