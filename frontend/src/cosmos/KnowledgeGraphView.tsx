import { useCallback, useEffect, useState } from "react";
import { getKGFacts } from "../lib/api";
import type { KGFact } from "../lib/types";
import KnowledgeGraphSection from "../pages/learning/KnowledgeGraphSection";

const PAGE = 100;

/** Knowledge region — the full graph tool: list/visual toggle, entity search,
 *  hop depth + fact-count controls, click a node for its facts/dossier. Wraps
 *  the proven KnowledgeGraphSection with fact pagination, defaulting to the
 *  graph visual. */
export default function KnowledgeGraphView() {
  const [facts, setFacts] = useState<KGFact[]>([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState("");
  const [hasMore, setHasMore] = useState(false);
  const [offset, setOffset] = useState(0);

  const load = useCallback(async (off: number, q: string) => {
    setLoading(true);
    try {
      const f = await getKGFacts(PAGE, off, q);
      setFacts((prev) => (off === 0 ? f : [...prev, ...f]));
      setHasMore(f.length === PAGE);
      setOffset(off + f.length);
    } catch {
      if (off === 0) setFacts([]);
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => { load(0, ""); }, [load]);

  return (
    <div className="flex h-full flex-col p-4 sm:p-5">
      <KnowledgeGraphSection
        initialView="graph"
        graphHeight="fill"
        facts={facts}
        loading={loading}
        search={search}
        hasMore={hasMore}
        onSearchChange={setSearch}
        onSearch={() => { setOffset(0); load(0, search); }}
        onLoadMore={() => load(offset, search)}
        onFactDeleted={() => { setOffset(0); load(0, search); }}
      />
    </div>
  );
}
