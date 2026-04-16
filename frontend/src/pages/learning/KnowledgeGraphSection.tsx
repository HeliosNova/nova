import { useState, useEffect, useCallback } from "react";
import { Search, Database, Trash2, Network, List, GitBranch } from "lucide-react";
import { toast } from "sonner";
import { Button, EmptyState, Skeleton, FormInput, FormSelect, ResponsiveTable, StatCard, ConfirmDialog } from "../../components/ui";
import type { Column } from "../../components/ui/ResponsiveTable";
import { formatDate, pct } from "../../lib/utils";
import { getKGGraph, getKGStats, deleteKGFact, searchMonitorResults, searchMessages } from "../../lib/api";
import type { KGFact, KGGraphData, KGStats, KGGraphNode } from "../../lib/types";
import KGGraph from "../../components/KGGraph";
import type { EntityInfo } from "../../components/KGGraph";

type ViewMode = "list" | "graph";

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
  onFactDeleted?: () => void;
}

export default function KnowledgeGraphSection({
  facts,
  loading,
  search,
  hasMore,
  onSearchChange,
  onSearch,
  onLoadMore,
  onFactDeleted,
}: Props) {
  const [viewMode, setViewMode] = useState<ViewMode>("list");
  const [graphData, setGraphData] = useState<KGGraphData>({ nodes: [], links: [] });
  const [graphLoading, setGraphLoading] = useState(false);
  const [graphEntity, setGraphEntity] = useState("");
  const [graphHops, setGraphHops] = useState(2);
  const [graphLimit, setGraphLimit] = useState(200);
  const [stats, setStats] = useState<KGStats | null>(null);
  const [deletingFactId, setDeletingFactId] = useState<number | null>(null);
  const [entityInfo, setEntityInfo] = useState<EntityInfo>({ monitors: [], msgs: [], loading: false });

  // Load stats on mount and after changes
  useEffect(() => {
    getKGStats().then(setStats).catch(() => {});
  }, [facts.length]);

  // Fetch entity content (monitor results + conversations)
  const fetchEntityInfo = useCallback(async (entity: string) => {
    setEntityInfo({ monitors: [], msgs: [], loading: true });
    const [monitorHits, msgs] = await Promise.all([
      searchMonitorResults(entity, 8).catch(() => []),
      searchMessages(entity).catch(() => []),
    ]);
    setEntityInfo({
      monitors: Array.isArray(monitorHits) ? monitorHits.slice(0, 8) : [],
      msgs: Array.isArray(msgs) ? msgs.slice(0, 8) : [],
      loading: false,
    });
  }, []);

  const loadGraph = useCallback(async (entity?: string, hops?: number, limit?: number) => {
    setGraphLoading(true);
    try {
      const data = await getKGGraph(entity, hops ?? graphHops, limit ?? graphLimit);
      setGraphData(data);
    } catch {
      toast.error("Failed to load graph data");
    } finally {
      setGraphLoading(false);
    }
  }, [graphHops, graphLimit]);

  // Load graph on first switch to graph view
  useEffect(() => {
    if (viewMode === "graph" && graphData.nodes.length === 0) {
      loadGraph(graphEntity || undefined);
    }
  }, [viewMode]);

  const handleGraphSearch = () => {
    loadGraph(graphEntity || undefined);
    if (graphEntity) fetchEntityInfo(graphEntity);
  };

  const handleNodeClick = (node: KGGraphNode) => {
    setGraphEntity(node.label);
    loadGraph(node.label);
    fetchEntityInfo(node.label);
  };

  // When switching to graph from list, carry search term over
  const handleViewChange = (mode: ViewMode) => {
    if (mode === "graph" && viewMode === "list" && search) {
      setGraphEntity(search);
    }
    setViewMode(mode);
  };

  const handleDeleteFact = async () => {
    if (deletingFactId === null) return;
    try {
      await deleteKGFact(deletingFactId);
      toast.success("Fact deleted");
      setDeletingFactId(null);
      onFactDeleted?.();
    } catch {
      toast.error("Failed to delete fact");
      setDeletingFactId(null);
    }
  };

  return (
    <section className="animate-fade-in">
      {/* Stats bar */}
      {stats && (
        <div className="mb-4 grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard label="Total Facts" value={stats.total_facts} />
          <StatCard label="Current" value={stats.current_facts} />
          <StatCard label="Entities" value={stats.unique_entities} />
          <StatCard label="Predicates" value={stats.unique_predicates} />
        </div>
      )}

      {/* View mode toggle + search */}
      <div className="mb-4 flex flex-col md:flex-row gap-2">
        <div className="flex gap-1 shrink-0">
          <button
            onClick={() => handleViewChange("list")}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
              viewMode === "list"
                ? "bg-nova-accent/15 text-nova-accent border border-nova-accent/30"
                : "text-nova-text-dim hover:text-nova-text hover:bg-nova-border/40 border border-transparent"
            }`}
          >
            <List size={14} /> List
          </button>
          <button
            onClick={() => handleViewChange("graph")}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
              viewMode === "graph"
                ? "bg-nova-accent/15 text-nova-accent border border-nova-accent/30"
                : "text-nova-text-dim hover:text-nova-text hover:bg-nova-border/40 border border-transparent"
            }`}
          >
            <Network size={14} /> Graph
          </button>
        </div>

        <div className="flex flex-1 gap-2">
          <FormInput
            value={viewMode === "graph" ? graphEntity : search}
            onChange={(e) =>
              viewMode === "graph"
                ? setGraphEntity(e.target.value)
                : onSearchChange(e.target.value)
            }
            onKeyDown={(e) =>
              e.key === "Enter" &&
              (viewMode === "graph" ? handleGraphSearch() : onSearch())
            }
            placeholder={
              viewMode === "graph"
                ? "Entity name (empty = top connected entities)..."
                : "Search knowledge graph facts..."
            }
            icon={<Search size={14} />}
            className="flex-1"
          />

          {/* Graph controls — hop depth + node limit */}
          {viewMode === "graph" && (
            <>
              <FormSelect
                value={String(graphHops)}
                onChange={(e) => {
                  const h = Number(e.target.value);
                  setGraphHops(h);
                  loadGraph(graphEntity || undefined, h, graphLimit);
                }}
                options={[
                  { value: "1", label: "1 hop" },
                  { value: "2", label: "2 hops" },
                  { value: "3", label: "3 hops" },
                ]}
              />
              <FormSelect
                value={String(graphLimit)}
                onChange={(e) => {
                  const l = Number(e.target.value);
                  setGraphLimit(l);
                  loadGraph(graphEntity || undefined, graphHops, l);
                }}
                options={[
                  { value: "100", label: "100 facts" },
                  { value: "200", label: "200 facts" },
                  { value: "500", label: "500 facts" },
                  { value: "1000", label: "1,000 facts" },
                  { value: "2000", label: "All" },
                ]}
              />
            </>
          )}

          <Button
            onClick={viewMode === "graph" ? handleGraphSearch : onSearch}
            loading={viewMode === "graph" ? graphLoading : loading && facts.length === 0}
          >
            {viewMode === "graph" ? "Explore" : "Search"}
          </Button>
        </div>
      </div>

      {/* Graph view */}
      {viewMode === "graph" && (
        <KGGraph
          graphData={graphData}
          onNodeClick={handleNodeClick}
          selectedEntity={graphEntity}
          entityInfo={entityInfo}
          loading={graphLoading}
          height={520}
        />
      )}

      {/* List view */}
      {viewMode === "list" && (
        <>
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
                renderRowSuffix={(fact) => (
                  <button
                    onClick={() => setDeletingFactId(fact.id)}
                    className="rounded p-1 text-nova-text-dim hover:text-nova-error hover:bg-nova-error/10 transition-colors"
                    title="Delete fact"
                  >
                    <Trash2 size={13} />
                  </button>
                )}
                headerSuffix=""
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
        </>
      )}

      {/* Delete confirmation */}
      {deletingFactId !== null && (
        <ConfirmDialog
          message="Delete this knowledge graph fact? This cannot be undone."
          onConfirm={handleDeleteFact}
          onCancel={() => setDeletingFactId(null)}
        />
      )}
    </section>
  );
}
