import { Search, Database } from "lucide-react";
import { Button, EmptyState, Skeleton, FormInput, ResponsiveTable } from "../../components/ui";
import type { Column } from "../../components/ui/ResponsiveTable";
import { formatDate, pct } from "../../lib/utils";
import type { KGFact } from "../../lib/types";

const kgColumns: Column<KGFact>[] = [
  {
    label: "Subject",
    accessor: (f) => <span className="font-medium">{f.subject}</span>,
    className: "max-w-[150px] truncate",
  },
  {
    label: "Predicate",
    accessor: (f) => <span className="text-nova-accent">{f.predicate}</span>,
    className: "max-w-[120px] truncate",
  },
  {
    label: "Object",
    accessor: (f) => <span className="text-nova-text-dim">{f.object}</span>,
    className: "max-w-[200px] truncate",
  },
  {
    label: "Confidence",
    accessor: (f) => (
      <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
        f.confidence >= 0.8
          ? "bg-nova-success/20 text-nova-success"
          : f.confidence >= 0.5
            ? "bg-nova-warning/20 text-nova-warning"
            : "bg-nova-error/20 text-nova-error"
      }`}>
        {pct(f.confidence)}
      </span>
    ),
    className: "text-center",
  },
  {
    label: "Source",
    accessor: (f) => <span className="text-xs text-nova-text-dim">{f.source}</span>,
    className: "max-w-[100px] truncate",
    hideOnMobile: true,
  },
  {
    label: "Valid From",
    accessor: (f) => <span className="text-xs text-nova-text-dim">{f.valid_from ? formatDate(f.valid_from) : "\u2014"}</span>,
    hideOnMobile: true,
  },
  {
    label: "Valid To",
    accessor: (f) =>
      f.valid_to === null || f.valid_to === undefined ? (
        <span className="rounded px-1.5 py-0.5 text-[10px] font-medium bg-nova-success/20 text-nova-success">current</span>
      ) : (
        <span className="text-xs text-nova-text-dim">{formatDate(f.valid_to)}</span>
      ),
    hideOnMobile: true,
  },
  {
    label: "Created",
    accessor: (f) => <span className="text-xs text-nova-text-dim">{formatDate(f.created_at)}</span>,
    hideOnMobile: true,
  },
];

interface Props {
  facts: KGFact[];
  loading: boolean;
  search: string;
  hasMore: boolean;
  onSearchChange: (value: string) => void;
  onSearch: () => void;
  onLoadMore: () => void;
}

export default function KnowledgeGraphSection({
  facts,
  loading,
  search,
  hasMore,
  onSearchChange,
  onSearch,
  onLoadMore,
}: Props) {
  return (
    <section>
      <div className="mb-4 flex gap-2">
        <FormInput
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onSearch()}
          placeholder="Search knowledge graph facts..."
          icon={<Search size={14} />}
          className="flex-1"
        />
        <Button onClick={onSearch} loading={loading && facts.length === 0}>
          Search
        </Button>
      </div>

      {loading && facts.length === 0 ? (
        <Skeleton lines={6} />
      ) : facts.length === 0 ? (
        <EmptyState
          icon={<Database size={40} strokeWidth={1.5} />}
          title="No knowledge graph facts found."
          description={search ? "Try a different search term." : "Facts are extracted from monitors, conversations, and domain studies."}
        />
      ) : (
        <>
          <div className="mb-2 text-xs text-nova-text-dim">
            Showing {facts.length} fact{facts.length !== 1 ? "s" : ""}
            {search && ` matching "${search}"`}
          </div>

          <ResponsiveTable<KGFact>
            columns={kgColumns}
            data={facts}
            keyFn={(fact) => fact.id}
          />

          {hasMore && (
            <div className="mt-4 flex justify-center">
              <Button
                variant="secondary"
                size="sm"
                onClick={onLoadMore}
                loading={loading}
              >
                Load More
              </Button>
            </div>
          )}
        </>
      )}
    </section>
  );
}
