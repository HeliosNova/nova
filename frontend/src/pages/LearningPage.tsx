import { useEffect, useState, useCallback, useRef } from "react";
import { GraduationCap, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { useLearningStore } from "../lib/store";
import {
  getLearningMetrics, getLessons, getSkills, deleteLesson, deleteSkill, toggleSkill,
  getReflexions, getTrainingData, getFinetuneStatus, getFinetuneHistory,
  getTrainingDataStatsDetailed, exportTrainingData, getConfigSummary, triggerFinetune,
  getKGFacts, getCuriosityQueue, bulkDeleteLessons,
} from "../lib/api";
import { formatDate } from "../lib/utils";
import type { ReflexionInfo, TrainingDataEntry, FinetuneStatus, FinetuneHistoryRun, TrainingDataStatsDetailed, KGFact, CuriosityItem } from "../lib/types";
import { PageHeader, Tabs, StatCard, ConfirmDialog, Skeleton } from "../components/ui";
import { LessonsSection, SkillsSection, ReflexionsSection, TrainingDataSection, KnowledgeGraphSection, CuriositySection } from "./learning";

type Section = "lessons" | "skills" | "reflexions" | "training" | "knowledge" | "curiosity";
type SortField = "confidence" | "times_retrieved" | "created_at";
type SortDir = "asc" | "desc";
const PAGE = 50;
const safe = <T,>(v: unknown): T[] => (Array.isArray(v) ? v : []);

export default function LearningPage() {
  const store = useLearningStore();
  const [confirm, setConfirm] = useState<{ type: "lesson" | "skill" | "bulk-lessons"; id: number; ids?: number[] } | null>(null);
  const [activeSection, setActiveSection] = useState<Section>("lessons");
  const [reflexions, setReflexions] = useState<ReflexionInfo[]>([]);
  const [reflexionsLoading, setReflexionsLoading] = useState(false);
  const [trainingData, setTrainingData] = useState<TrainingDataEntry[]>([]);
  const [finetuneStatus, setFinetuneStatus] = useState<FinetuneStatus | null>(null);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [finetuneRunning, setFinetuneRunning] = useState(false);
  const [trainingOffset, setTrainingOffset] = useState(0);
  const [trainingHasMore, setTrainingHasMore] = useState(true);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [togglingSkillId, setTogglingSkillId] = useState<number | null>(null);
  const [sortField, setSortField] = useState<SortField>("confidence");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [metricsRefreshing, setMetricsRefreshing] = useState(false);
  const [selectedLessons, setSelectedLessons] = useState<Set<number>>(new Set());
  const [finetuneHistory, setFinetuneHistory] = useState<FinetuneHistoryRun[]>([]);
  const [trainingStats, setTrainingStats] = useState<TrainingDataStatsDetailed | null>(null);
  const [curiosityItems, setCuriosityItems] = useState<CuriosityItem[]>([]);
  const [curiosityLoading, setCuriosityLoading] = useState(false);
  const [autoSkills, setAutoSkills] = useState(false);
  const [kgFacts, setKgFacts] = useState<KGFact[]>([]);
  const [kgLoading, setKgLoading] = useState(false);
  const [kgSearch, setKgSearch] = useState("");
  const [kgOffset, setKgOffset] = useState(0);
  const [kgHasMore, setKgHasMore] = useState(true);

  useEffect(() => { getConfigSummary().then((c) => setAutoSkills(c.ENABLE_AUTO_SKILL_CREATION)).catch(() => {}); }, []);

  useEffect(() => {
    store.setLoading(true);
    Promise.all([
      getLearningMetrics().then(store.setMetrics).catch(() => {}),
      getLessons().then((v) => store.setLessons(safe(v))).catch(() => store.setLessons([])),
      getSkills().then((v) => store.setSkills(safe(v))).catch(() => store.setSkills([])),
    ]).finally(() => store.setLoading(false));
  }, []);

  // Track a refresh counter to force re-fetch after mutations
  const [refreshKey, setRefreshKey] = useState(0);
  const triggerRefresh = useCallback(() => setRefreshKey((k) => k + 1), []);

  useEffect(() => {
    if (activeSection === "lessons") {
      store.setLoading(true);
      Promise.all([
        getLearningMetrics().then(store.setMetrics).catch(() => {}),
        getLessons().then((v) => store.setLessons(safe(v))).catch(() => store.setLessons([])),
      ]).finally(() => store.setLoading(false));
    }
    if (activeSection === "skills") {
      getSkills().then((v) => store.setSkills(safe(v))).catch(() => store.setSkills([]));
    }
    if (activeSection === "reflexions") {
      setReflexionsLoading(true);
      getReflexions().then((v) => setReflexions(safe(v))).catch(() => setReflexions([])).finally(() => setReflexionsLoading(false));
    }
    if (activeSection === "training") {
      setTrainingLoading(true);
      Promise.all([
        getTrainingData(PAGE, 0).then((v) => { const a = safe<TrainingDataEntry>(v); setTrainingData(a); setTrainingOffset(a.length); setTrainingHasMore(a.length === PAGE); }).catch(() => setTrainingData([])),
        getFinetuneStatus().then(setFinetuneStatus).catch(() => {}),
        getFinetuneHistory().then((v) => setFinetuneHistory(safe(v))).catch(() => {}),
        getTrainingDataStatsDetailed().then(setTrainingStats).catch(() => {}),
      ]).finally(() => setTrainingLoading(false));
    }
    if (activeSection === "knowledge") loadKG(0, kgSearch);
    if (activeSection === "curiosity") {
      setCuriosityLoading(true);
      getCuriosityQueue().then((v) => setCuriosityItems(safe(v))).catch(() => setCuriosityItems([])).finally(() => setCuriosityLoading(false));
    }
  }, [activeSection, refreshKey]);

  useEffect(() => {
    const h = () => { if (document.visibilityState === "visible") refreshMetrics(); };
    document.addEventListener("visibilitychange", h);
    return () => document.removeEventListener("visibilitychange", h);
  }, []);
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const refreshMetrics = useCallback(async () => {
    setMetricsRefreshing(true);
    try { store.setMetrics(await getLearningMetrics()); } catch {} finally { setMetricsRefreshing(false); }
  }, []);

  const loadKG = useCallback(async (offset: number, search: string) => {
    setKgLoading(true);
    try {
      const f = await getKGFacts(PAGE, offset, search);
      offset === 0 ? setKgFacts(f) : setKgFacts((p) => [...p, ...f]);
      setKgHasMore(f.length === PAGE); setKgOffset(offset + f.length);
    } catch { if (offset === 0) setKgFacts([]); } finally { setKgLoading(false); }
  }, []);

  const handleDeleteConfirm = async () => {
    if (!confirm) return;
    try {
      if (confirm.type === "bulk-lessons" && confirm.ids) {
        await bulkDeleteLessons(confirm.ids); confirm.ids.forEach((id) => store.removeLesson(id));
        setSelectedLessons(new Set()); toast.success(`${confirm.ids.length} lesson${confirm.ids.length !== 1 ? "s" : ""} deleted`);
      } else if (confirm.type === "lesson") {
        await deleteLesson(confirm.id); store.removeLesson(confirm.id);
        setSelectedLessons((p) => { const n = new Set(p); n.delete(confirm.id); return n; }); toast.success("Lesson deleted");
      } else { await deleteSkill(confirm.id); store.removeSkill(confirm.id); toast.success("Skill deleted"); }
      triggerRefresh();
    } catch { toast.error(`Failed to delete ${confirm.type === "bulk-lessons" ? "lessons" : confirm.type}`); }
    setConfirm(null);
  };

  const handleToggle = async (id: number, enabled: boolean) => {
    setTogglingSkillId(id);
    try { await toggleSkill(id, !enabled); store.updateSkillEnabled(id, !enabled); triggerRefresh(); } catch { toast.error("Failed to toggle skill"); } finally { setTogglingSkillId(null); }
  };

  const handleExportTraining = async () => {
    try {
      const url = URL.createObjectURL(await exportTrainingData());
      Object.assign(document.createElement("a"), { href: url, download: "training_data_dpo.json" }).click();
      URL.revokeObjectURL(url); toast.success("Training data exported");
    } catch { toast.error("Failed to export training data"); }
  };

  const handleTriggerFinetune = async () => {
    setFinetuneRunning(true);
    const stopPoll = () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; } setFinetuneRunning(false); };
    try {
      toast.success((await triggerFinetune()).message || "Fine-tuning started successfully");
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try { const s = await getFinetuneStatus(); setFinetuneStatus(s); if (!s || (s as FinetuneStatus & { status?: string }).status !== "running") stopPoll(); }
        catch { stopPoll(); }
      }, 10000);
    } catch (e) { toast.error(`Fine-tuning failed: ${(e as Error).message}`); stopPoll(); }
  };

  const handleTrainingLoadMore = useCallback(async () => {
    setTrainingLoading(true);
    try {
      const a = safe<TrainingDataEntry>(await getTrainingData(PAGE, trainingOffset));
      setTrainingData((p) => [...p, ...a]); setTrainingOffset((p) => p + a.length); setTrainingHasMore(a.length === PAGE);
    } catch {} finally { setTrainingLoading(false); }
  }, [trainingOffset]);

  const handleSort = (f: SortField) => { f === sortField ? setSortDir((d) => d === "desc" ? "asc" : "desc") : (setSortField(f), setSortDir("desc")); };
  const toggleSelectAll = () => { store.lessons.length > 0 && selectedLessons.size === store.lessons.length ? setSelectedLessons(new Set()) : setSelectedLessons(new Set(store.lessons.map((l) => l.id))); };
  const toggleSelectLesson = (id: number) => { setSelectedLessons((p) => { const n = new Set(p); n.has(id) ? n.delete(id) : n.add(id); return n; }); };

  if (store.loading) return <div className="flex-1 overflow-y-auto"><div className="mx-auto max-w-4xl w-full px-4 md:px-6 py-6"><Skeleton lines={6} /></div></div>;

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl w-full px-4 md:px-6 py-6">
        <PageHeader icon={<GraduationCap size={22} />} title="Learning Dashboard" />
        {store.metrics && (
          <div className="mb-8">
            <div className="mb-2 flex justify-end">
              <button onClick={refreshMetrics} disabled={metricsRefreshing} className="flex items-center gap-1.5 text-xs text-nova-text-dim hover:text-nova-text transition-colors disabled:opacity-40" title="Refresh metrics">
                <RefreshCw size={12} className={metricsRefreshing ? "animate-spin" : ""} /> Refresh
              </button>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard label="Lessons" value={store.metrics.total_lessons} />
              <StatCard label="Skills" value={store.metrics.total_skills} />
              <StatCard label="Corrections" value={store.metrics.total_corrections} />
              <StatCard label="Training Examples" value={store.metrics.training_examples} sub={store.metrics.last_correction_date ? `Last: ${formatDate(store.metrics.last_correction_date)}` : undefined} />
            </div>
          </div>
        )}
        <Tabs tabs={[
          { id: "lessons", label: `Lessons (${store.lessons.length})` },
          { id: "skills", label: `Skills (${store.skills.length})`, badge: autoSkills },
          { id: "reflexions", label: "Reflexions" }, { id: "training", label: "Training Data" },
          { id: "knowledge", label: "Knowledge Graph" }, { id: "curiosity", label: "Curiosity" },
        ]} active={activeSection} onChange={(id) => setActiveSection(id as Section)} />

        {activeSection === "lessons" && <LessonsSection lessons={store.lessons} sortField={sortField} sortDir={sortDir} selectedLessons={selectedLessons} onSort={handleSort} onToggleSelectAll={toggleSelectAll} onToggleSelect={toggleSelectLesson} onDelete={(id) => setConfirm({ type: "lesson", id })} onBulkDelete={(ids) => setConfirm({ type: "bulk-lessons", id: 0, ids })} />}
        {activeSection === "skills" && <SkillsSection skills={store.skills} togglingSkillId={togglingSkillId} onToggle={handleToggle} onDelete={(id) => setConfirm({ type: "skill", id })} />}
        {activeSection === "reflexions" && <ReflexionsSection reflexions={reflexions} loading={reflexionsLoading} />}
        {activeSection === "training" && <TrainingDataSection trainingData={trainingData} trainingStats={trainingStats} finetuneStatus={finetuneStatus} finetuneHistory={finetuneHistory} loading={trainingLoading} finetuneRunning={finetuneRunning} hasMore={trainingHasMore} onLoadMore={handleTrainingLoadMore} onExport={handleExportTraining} onTriggerFinetune={handleTriggerFinetune} />}
        {activeSection === "knowledge" && <KnowledgeGraphSection facts={kgFacts} loading={kgLoading} search={kgSearch} hasMore={kgHasMore} onSearchChange={setKgSearch} onSearch={() => { setKgOffset(0); setKgFacts([]); loadKG(0, kgSearch); }} onLoadMore={() => loadKG(kgOffset, kgSearch)} />}
        {activeSection === "curiosity" && <CuriositySection items={curiosityItems} loading={curiosityLoading} />}

        {confirm && <ConfirmDialog message={confirm.type === "bulk-lessons" ? `Delete ${confirm.ids?.length} selected lesson${(confirm.ids?.length ?? 0) !== 1 ? "s" : ""}? This cannot be undone.` : `Delete this ${confirm.type}? This cannot be undone.`} onConfirm={handleDeleteConfirm} onCancel={() => setConfirm(null)} />}
      </div>
    </div>
  );
}
