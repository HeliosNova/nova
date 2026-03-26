import type { ReactNode } from "react";
import { GraduationCap, ArrowUp, ArrowDown } from "lucide-react";
import { Button, EmptyState, ResponsiveTable } from "../../components/ui";
import { formatDate, pct } from "../../lib/utils";
import type { LessonInfo } from "../../lib/types";
import type { Column } from "../../components/ui/ResponsiveTable";

type SortField = "confidence" | "times_retrieved" | "created_at";
type SortDir = "asc" | "desc";

interface Props {
  lessons: LessonInfo[];
  sortField: SortField;
  sortDir: SortDir;
  selectedLessons: Set<number>;
  onSort: (field: SortField) => void;
  onToggleSelectAll: () => void;
  onToggleSelect: (id: number) => void;
  onDelete: (id: number) => void;
  onBulkDelete: (ids: number[]) => void;
}

export default function LessonsSection({
  lessons,
  sortField,
  sortDir,
  selectedLessons,
  onSort,
  onToggleSelectAll,
  onToggleSelect,
  onDelete,
  onBulkDelete,
}: Props) {
  const allSelected = lessons.length > 0 && selectedLessons.size === lessons.length;
  const someSelected = selectedLessons.size > 0;

  const sortedLessons = [...lessons].sort((a, b) => {
    let cmp = 0;
    if (sortField === "confidence") cmp = a.confidence - b.confidence;
    else if (sortField === "times_retrieved") cmp = a.times_retrieved - b.times_retrieved;
    else if (sortField === "created_at") cmp = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
    return sortDir === "desc" ? -cmp : cmp;
  });

  const SortArrow = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null;
    return sortDir === "desc" ? <ArrowDown size={12} className="inline ml-0.5" /> : <ArrowUp size={12} className="inline ml-0.5" />;
  };

  const columns: Column<LessonInfo>[] = [
    {
      label: "Topic",
      accessor: (l) => <span className="font-medium">{l.topic}</span>,
      className: "max-w-[200px] truncate",
    },
    {
      label: "Correct Answer",
      accessor: (l) => <span className="text-nova-text-dim">{l.correct_answer}</span>,
      className: "max-w-[250px] truncate",
    },
    {
      label: "Confidence",
      accessor: (l) => pct(l.confidence),
      className: "text-center cursor-pointer select-none hover:text-nova-text transition-colors",
    },
    {
      label: "Times Used",
      accessor: (l) => String(l.times_retrieved),
      className: "text-center cursor-pointer select-none hover:text-nova-text transition-colors",
    },
    {
      label: "Created",
      accessor: (l) => <span className="text-xs text-nova-text-dim">{formatDate(l.created_at)}</span>,
      className: "text-center cursor-pointer select-none hover:text-nova-text transition-colors",
    },
  ];

  if (lessons.length === 0) {
    return (
      <section>
        <EmptyState
          icon={<GraduationCap size={40} strokeWidth={1.5} />}
          title="No lessons learned yet."
          description="Start chatting with Nova and correct it when it's wrong — lessons are created automatically from corrections."
        />
      </section>
    );
  }

  return (
    <section>
      {/* Bulk delete bar */}
      {someSelected && (
        <div className="mb-3 flex items-center gap-3 rounded-lg border border-nova-border bg-nova-surface px-3 py-2">
          <span className="text-sm text-nova-text-dim">
            {selectedLessons.size} selected
          </span>
          <Button
            variant="danger"
            size="sm"
            onClick={() => onBulkDelete(Array.from(selectedLessons))}
          >
            Delete selected ({selectedLessons.size})
          </Button>
        </div>
      )}

      {/* Desktop: custom sortable table with checkboxes */}
      <div className="hidden md:block overflow-x-auto rounded-lg border border-nova-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-nova-border bg-nova-surface text-left text-xs text-nova-text-dim">
              <th className="px-3 py-2 w-8">
                <input
                  type="checkbox"
                  checked={allSelected}
                  onChange={onToggleSelectAll}
                  className="accent-nova-accent"
                />
              </th>
              <th className="px-3 py-2">Topic</th>
              <th className="px-3 py-2">Correct Answer</th>
              <th
                className="px-3 py-2 text-center cursor-pointer select-none hover:text-nova-text transition-colors"
                onClick={() => onSort("confidence")}
              >
                Confidence <SortArrow field="confidence" />
              </th>
              <th
                className="px-3 py-2 text-center cursor-pointer select-none hover:text-nova-text transition-colors"
                onClick={() => onSort("times_retrieved")}
              >
                Retrieved <SortArrow field="times_retrieved" />
              </th>
              <th
                className="px-3 py-2 text-center cursor-pointer select-none hover:text-nova-text transition-colors"
                onClick={() => onSort("created_at")}
              >
                Created <SortArrow field="created_at" />
              </th>
              <th className="px-3 py-2 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {sortedLessons.map((l) => (
              <tr
                key={l.id}
                className="border-b border-nova-border last:border-0 hover:bg-nova-surface/50"
              >
                <td className="px-3 py-2">
                  <input
                    type="checkbox"
                    checked={selectedLessons.has(l.id)}
                    onChange={() => onToggleSelect(l.id)}
                    className="accent-nova-accent"
                  />
                </td>
                <td className="max-w-[200px] truncate px-3 py-2 font-medium">{l.topic}</td>
                <td className="max-w-[250px] truncate px-3 py-2 text-nova-text-dim">{l.correct_answer}</td>
                <td className="px-3 py-2 text-center">{pct(l.confidence)}</td>
                <td className="px-3 py-2 text-center">{l.times_retrieved}</td>
                <td className="px-3 py-2 text-center text-xs text-nova-text-dim">{formatDate(l.created_at)}</td>
                <td className="px-3 py-2 text-right">
                  <Button variant="ghost" size="sm" onClick={() => onDelete(l.id)}>
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
        <ResponsiveTable<LessonInfo>
          columns={columns}
          data={sortedLessons}
          keyFn={(l) => l.id}
          renderRowPrefix={(l) => (
            <input
              type="checkbox"
              checked={selectedLessons.has(l.id)}
              onChange={() => onToggleSelect(l.id)}
              className="accent-nova-accent"
            />
          )}
          renderRowSuffix={(l) => (
            <Button variant="ghost" size="sm" onClick={() => onDelete(l.id)}>
              Delete
            </Button>
          )}
        />
      </div>
    </section>
  );
}
