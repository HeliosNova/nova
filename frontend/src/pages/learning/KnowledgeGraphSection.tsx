import { useState, useEffect, useCallback } from "react";
import { Search, Database, Trash2, Network, List, GitBranch } from "lucide-react";
import { toast } from "sonner";
import { Button, EmptyState, Skeleton, FormInput, FormSelect, ResponsiveTable, StatCard, ConfirmDialog } from "../../components/ui";
import type { Column } from "../../components/ui/ResponsiveTable";
import { formatDate, pct } from "../../lib/utils";
import { getKGGraph, getKGStats, deleteKGFact, searchMonitorResults, searchMessages, getDossiers, getDossier } from "../../lib/api";
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
  initialView?: ViewMode;
  /** When "fill", the section flexes to its parent's height and the graph fills
   *  the remaining space (used in the full-height cosmos Knowledge region).
   *  A number keeps the legacy fixed-height layout (the Learning page). */
  graphHeight?: number | "fill";
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
  initialView = "list",
  graphHeight = 640,
}: Props) {
  const fill = graphHeight === "fill";
  const [viewMode, setViewMode] = useState<ViewMode>(initialView);
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

  // slug mirrors backend dossiers._slug (lowercase, non-alnum → dash).
  const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 80);

  // Pull the "Current understanding" prose out of a dossier body (the section
  // most useful in a compact panel), falling back to the body head.
  const understandingOf = (body: string): string => {
    const m = body.match(/##\s*Current understanding\s*\n([\s\S]*?)(?=\n##\s|\n#\s|$)/i);
    const txt = (m ? m[1] : body).trim();
    return txt.length > 900 ? txt.slice(0, 900).replace(/\s+\S*$/, "") + "…" : txt;
  };

  // Fetch entity content (dossier + monitor results + conversations)
  const fetchEntityInfo = useCallback(async (entity: string) => {
    setEntityInfo({ monitors: [], msgs: [], loading: true, dossier: null });
    const key = slug(entity);
    const [monitorHits, msgs, dossierList] = await Promise.all([
      searchMonitorResults(entity, 8).catch(() => []),
      searchMessages(entity).catch(() => []),
      getDossiers("entity").catch(() => []),
    ]);
    // Match an entity dossier by slug (dkey) or title, then load its body.
    let dossier: EntityInfo["dossier"] = null;
    const hit = (Array.isArray(dossierList) ? dossierList : []).find(
      (d) => d.dkey === key || d.title.toLowerCase() === entity.toLowerCase());
    if (hit) {
      try {
        const full = await getDossier(hit.id);
        dossier = { id: hit.id, title: hit.title, understanding: understandingOf(full.body || ""), updated_at: hit.updated_at };
      } catch { /* dossier is a bonus; ignore fetch failure */ }
    }
    setEntityInfo({
      monitors: Array.isArray(monitorHits) ? monitorHits.slice(0, 8) : [],
      msgs: Array.isArray(msgs) ? msgs.slice(0, 8) : [],
      loading: false,
      dossier,
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

  // Carry the search term across view switches (both directions) so the two
  // views can't silently show results for different queries.
  const handleViewChange = (mode: ViewMode) => {
    if (mode === "graph" && viewMode === "list" && search) {
      setGraphEntity(search);
    } else if (mode === "list" && viewMode === "graph" && graphEntity !== search) {
      onSearchChange(graphEntity);
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
    <section className={fill ? "flex min-h-0 flex-1 flex-col animate-fade-in" : "animate-fade-in"}>
      {/* Stats bar */}
      {stats && (
        <div className={`mb-4 grid grid-cols-2 md:grid-cols-4 gap-3 ${fill ? "shrink-0" : ""}`}>
          <StatCard label="Total Facts" value={stats.total_facts} />
          <StatCard label="Current" value={stats.current_facts} />
          <StatCard label="Entities" value={stats.unique_entities} />
          <StatCard label="Predicates" value={stats.unique_predicates} />
        </div>
      )}

      {/* View mode toggle + search */}
      <div className={`mb-4 flex flex-col md:flex-row gap-2 ${fill ? "shrink-0" : ""}`}>
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
        <div className={fill ? "min-h-0 flex-1" : ""}>
          <KGGraph
            graphData={graphData}
            onNodeClick={handleNodeClick}
            selectedEntity={graphEntity}
            entityInfo={entityInfo}
            loading={graphLoading}
            height={graphHeight}
          />
        </div>
      )}

      {/* List view */}
      {viewMode === "list" && (
        <div className={fill ? "min-h-0 flex-1 overflow-y-auto" : "contents"}>
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
        </div>
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
