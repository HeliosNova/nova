import { GraduationCap, Loader2 } from "lucide-react";
import { Button, EmptyState, ResponsiveTable } from "../../components/ui";
import { pct } from "../../lib/utils";
import type { SkillInfo } from "../../lib/types";
import type { Column } from "../../components/ui/ResponsiveTable";
import type { ReactNode } from "react";

interface Props {
  skills: SkillInfo[];
  togglingSkillId: number | null;
  onToggle: (id: number, enabled: boolean) => void;
  onDelete: (id: number) => void;
}

export default function SkillsSection({ skills, togglingSkillId, onToggle, onDelete }: Props) {
  if (skills.length === 0) {
    return (
      <section>
        <EmptyState
          icon={<GraduationCap size={40} strokeWidth={1.5} />}
          title="No skills acquired yet."
        />
      </section>
    );
  }

  const columns: Column<SkillInfo>[] = [
    {
      label: "Name",
      accessor: (s) => <span className="font-medium">{s.name}</span>,
    },
    {
      label: "Trigger",
      accessor: (s) => <span className="text-nova-text-dim">{s.trigger_pattern}</span>,
      className: "max-w-[200px] truncate",
    },
    {
      label: "Used",
      accessor: (s) => String(s.times_used),
      className: "text-center",
    },
    {
      label: "Success",
      accessor: (s) => pct(s.success_rate),
      className: "text-center",
    },
    {
      label: "Enabled",
      accessor: (s) => (
        <button
          onClick={() => onToggle(s.id, s.enabled)}
          disabled={togglingSkillId === s.id}
          className={`rounded px-2 py-0.5 text-xs font-medium transition-opacity ${
            togglingSkillId === s.id ? "opacity-40" : ""
          } ${
            s.enabled
              ? "bg-nova-success/20 text-nova-success"
              : "bg-nova-error/20 text-nova-error"
          }`}
        >
          {togglingSkillId === s.id ? (
            <Loader2 size={12} className="inline animate-spin" />
          ) : s.enabled ? "ON" : "OFF"}
        </button>
      ),
      className: "text-center",
    },
  ];

  return (
    <section>
      {/* Desktop: standard table */}
      <div className="hidden md:block overflow-x-auto rounded-lg border border-nova-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-nova-border bg-nova-surface text-left text-xs text-nova-text-dim">
              <th className="px-3 py-2">Name</th>
              <th className="px-3 py-2">Trigger</th>
              <th className="px-3 py-2 text-center">Used</th>
              <th className="px-3 py-2 text-center">Success</th>
              <th className="px-3 py-2 text-center">Enabled</th>
              <th className="px-3 py-2 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {skills.map((s) => (
              <tr
                key={s.id}
                className="border-b border-nova-border last:border-0 hover:bg-nova-surface/50"
              >
                <td className="px-3 py-2 font-medium">{s.name}</td>
                <td className="max-w-[200px] truncate px-3 py-2 text-nova-text-dim">{s.trigger_pattern}</td>
                <td className="px-3 py-2 text-center">{s.times_used}</td>
                <td className="px-3 py-2 text-center">{pct(s.success_rate)}</td>
                <td className="px-3 py-2 text-center">
                  <button
                    onClick={() => onToggle(s.id, s.enabled)}
                    disabled={togglingSkillId === s.id}
                    className={`rounded px-2 py-0.5 text-xs font-medium transition-opacity ${
                      togglingSkillId === s.id ? "opacity-40" : ""
                    } ${
                      s.enabled
                        ? "bg-nova-success/20 text-nova-success"
                        : "bg-nova-error/20 text-nova-error"
                    }`}
                  >
                    {togglingSkillId === s.id ? (
                      <Loader2 size={12} className="inline animate-spin" />
                    ) : s.enabled ? "ON" : "OFF"}
                  </button>
                </td>
                <td className="px-3 py-2 text-right">
                  <Button variant="ghost" size="sm" onClick={() => onDelete(s.id)}>
                    Delete
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Mobile: stacked cards via ResponsiveTable */}
      <div className="md:hidden">
        <ResponsiveTable<SkillInfo>
          columns={columns}
          data={skills}
          keyFn={(s) => s.id}
          renderRowSuffix={(s) => (
            <Button variant="ghost" size="sm" onClick={() => onDelete(s.id)}>
              Delete
            </Button>
          )}
        />
      </div>
    </section>
  );
}
