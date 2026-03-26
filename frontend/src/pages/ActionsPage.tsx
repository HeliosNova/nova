import { useEffect, useState } from "react";
import { Zap, ChevronDown, ChevronUp } from "lucide-react";
import { getActions } from "../lib/api";
import { formatDate } from "../lib/utils";
import type { ActionInfo } from "../lib/types";
import { PageHeader, FormSelect, EmptyState, Skeleton } from "../components/ui";

function ActionRow({ action }: { action: ActionInfo }) {
  const [expanded, setExpanded] = useState(false);
  const hasDetails = (action.params && action.params.length > 80) || (action.result && action.result.length > 100);

  return (
    <>
      <tr
        onClick={() => hasDetails && setExpanded(!expanded)}
        className={`border-b border-nova-border last:border-0 hover:bg-nova-surface/50 ${hasDetails ? "cursor-pointer" : ""}`}
      >
        <td className="whitespace-nowrap px-3 py-2 text-nova-text-dim">
          {formatDate(action.created_at)}
        </td>
        <td className="px-3 py-2 font-medium">{action.action_type}</td>
        <td className="max-w-[200px] truncate px-3 py-2 text-nova-text-dim">
          {(action.params || "—").slice(0, 80)}
        </td>
        <td className="px-3 py-2">
          <span
            className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
              action.success
                ? "bg-nova-success/20 text-nova-success"
                : "bg-nova-error/20 text-nova-error"
            }`}
          >
            {action.success ? "success" : "error"}
          </span>
        </td>
        <td className="max-w-[250px] px-3 py-2 text-nova-text-dim">
          <div className="flex items-center gap-1">
            <span className="truncate">{(action.result || "—").slice(0, 100)}</span>
            {hasDetails && (
              <span className="shrink-0 text-nova-text-dim">
                {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </span>
            )}
          </div>
        </td>
      </tr>
      {expanded && (
        <tr className="border-b border-nova-border bg-nova-bg/50">
          <td colSpan={5} className="px-4 py-3">
            <div className="space-y-2 text-xs">
              {action.params && (
                <div>
                  <span className="font-medium text-nova-text-dim">Params:</span>
                  <pre className="mt-1 whitespace-pre-wrap break-all rounded border border-nova-border bg-nova-bg p-2 text-nova-text-dim">
                    {action.params}
                  </pre>
                </div>
              )}
              {action.result && (
                <div>
                  <span className="font-medium text-nova-text-dim">Result:</span>
                  <pre className="mt-1 whitespace-pre-wrap break-all rounded border border-nova-border bg-nova-bg p-2 text-nova-text-dim">
                    {action.result}
                  </pre>
                </div>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export default function ActionsPage() {
  const [actions, setActions] = useState<ActionInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterType, setFilterType] = useState("");
  const [hours, setHours] = useState(24);

  const refresh = () => {
    setLoading(true);
    getActions(filterType || undefined, hours)
      .then((v) => setActions(Array.isArray(v) ? v : []))
      .catch(() => setActions([]))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    refresh();
  }, [filterType, hours]);

  const actionTypes = [...new Set(actions.map((a) => a.action_type))].sort();

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl w-full px-4 md:px-6 py-6">
        <PageHeader icon={<Zap size={22} />} title="Action Audit Log" />
        <p className="mb-4 -mt-2 text-xs text-nova-text-dim">
          System actions logged when Nova uses tools, sends alerts, runs monitors, or processes background tasks.
        </p>

        {/* Filters */}
        <div className="mb-4 flex items-center gap-3">
          <FormSelect
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            placeholder="All types"
            options={actionTypes.map((t) => ({ value: t, label: t }))}
          />
          <FormSelect
            value={String(hours)}
            onChange={(e) => setHours(Number(e.target.value))}
            options={[
              { value: "1", label: "Last hour" },
              { value: "6", label: "Last 6 hours" },
              { value: "24", label: "Last 24 hours" },
              { value: "72", label: "Last 3 days" },
              { value: "168", label: "Last week" },
            ]}
          />
          <span className="text-xs text-nova-text-dim">{actions.length} actions</span>
        </div>

        {/* Actions table */}
        {loading ? (
          <Skeleton lines={4} />
        ) : actions.length === 0 ? (
          <EmptyState
            icon={<Zap size={40} strokeWidth={1.5} />}
            title="No actions in this time range."
            description="Actions are logged when Nova uses tools, runs monitors, or processes background tasks. Try a longer time range."
          />
        ) : (
          <div className="overflow-x-auto rounded-lg border border-nova-border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-nova-border bg-nova-surface text-left text-xs text-nova-text-dim">
                  <th className="px-3 py-2">Time</th>
                  <th className="px-3 py-2">Type</th>
                  <th className="px-3 py-2">Params</th>
                  <th className="px-3 py-2">Status</th>
                  <th className="px-3 py-2">Result</th>
                </tr>
              </thead>
              <tbody>
                {actions.map((a) => (
                  <ActionRow key={a.id} action={a} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
