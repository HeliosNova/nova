import { useEffect, useState, useCallback, useRef } from "react";
import { GraduationCap, Download, Play, Search, Database, RefreshCw, ArrowUp, ArrowDown, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { useLearningStore } from "../lib/store";
import {
  getLearningMetrics,
  getLessons,
  getSkills,
  deleteLesson,
  deleteSkill,
  toggleSkill,
  getReflexions,
  getTrainingData,
  getFinetuneStatus,
  getFinetuneHistory,
  getTrainingDataStatsDetailed,
  exportTrainingData,
  getConfigSummary,
  triggerFinetune,
  getKGFacts,
  getCuriosityQueue,
  bulkDeleteLessons,
} from "../lib/api";
import { formatDate, pct } from "../lib/utils";
import type { ReflexionInfo, TrainingDataEntry, FinetuneStatus, FinetuneHistoryRun, TrainingDataStatsDetailed, KGFact, CuriosityItem } from "../lib/types";
import {
  PageHeader,
  Card,
  Button,
  Tabs,
  StatCard,
  EmptyState,
  ConfirmDialog,
  Skeleton,
  FormInput,
} from "../components/ui";

type Section = "lessons" | "skills" | "reflexions" | "training" | "knowledge" | "curiosity";
type SortField = "confidence" | "times_retrieved" | "created_at";
type SortDir = "asc" | "desc";

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

  // MED-2: Training data pagination
  const [trainingOffset, setTrainingOffset] = useState(0);
  const [trainingHasMore, setTrainingHasMore] = useState(true);
  const TRAINING_PAGE_SIZE = 50;

  // MED-3: Finetune status polling
  const finetunePollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // MED-5: Skill toggle loading state
  const [togglingSkillId, setTogglingSkillId] = useState<number | null>(null);

  // MED-7: Lessons table sorting
  const [sortField, setSortField] = useState<SortField>("confidence");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  // MED-8: Metrics refresh
  const [metricsRefreshing, setMetricsRefreshing] = useState(false);

  // MED-11: Bulk delete lessons
  const [selectedLessons, setSelectedLessons] = useState<Set<number>>(new Set());

  // HIGH-1: Finetune history
  const [finetuneHistory, setFinetuneHistory] = useState<FinetuneHistoryRun[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // HIGH-2: Training data stats
  const [trainingStats, setTrainingStats] = useState<TrainingDataStatsDetailed | null>(null);

  // HIGH-3: Curiosity queue
  const [curiosityItems, setCuriosityItems] = useState<CuriosityItem[]>([]);
  const [curiosityLoading, setCuriosityLoading] = useState(false);

  const [autoSkills, setAutoSkills] = useState(false);

  // KG state
  const [kgFacts, setKgFacts] = useState<KGFact[]>([]);
  const [kgLoading, setKgLoading] = useState(false);
  const [kgSearch, setKgSearch] = useState("");
  const [kgOffset, setKgOffset] = useState(0);
  const [kgHasMore, setKgHasMore] = useState(true);
  const KG_PAGE_SIZE = 50;

  useEffect(() => {
    getConfigSummary()
      .then((c) => setAutoSkills(c.ENABLE_AUTO_SKILL_CREATION))
      .catch(() => {});
  }, []);

  useEffect(() => {
    store.setLoading(true);
    Promise.all([
      getLearningMetrics().then(store.setMetrics).catch(() => {}),
      getLessons().then((v) => store.setLessons(Array.isArray(v) ? v : [])).catch(() => store.setLessons([])),
      getSkills().then((v) => store.setSkills(Array.isArray(v) ? v : [])).catch(() => store.setSkills([])),
    ]).finally(() => store.setLoading(false));
  }, []);

  useEffect(() => {
    if (activeSection === "reflexions" && reflexions.length === 0) {
      setReflexionsLoading(true);
      getReflexions()
        .then((v) => setReflexions(Array.isArray(v) ? v : []))
        .catch(() => setReflexions([]))
        .finally(() => setReflexionsLoading(false));
    }
    if (activeSection === "training" && trainingData.length === 0) {
      setTrainingLoading(true);
      Promise.all([
        getTrainingData(TRAINING_PAGE_SIZE, 0).then((v) => {
          const arr = Array.isArray(v) ? v : [];
          setTrainingData(arr);
          setTrainingOffset(arr.length);
          setTrainingHasMore(arr.length === TRAINING_PAGE_SIZE);
        }).catch(() => setTrainingData([])),
        getFinetuneStatus().then(setFinetuneStatus).catch(() => {}),
        getFinetuneHistory().then((v) => setFinetuneHistory(Array.isArray(v) ? v : [])).catch(() => {}),
        getTrainingDataStatsDetailed().then(setTrainingStats).catch(() => {}),
      ]).finally(() => setTrainingLoading(false));
    }
    if (activeSection === "knowledge" && kgFacts.length === 0) {
      loadKGFacts(0, "");
    }
    if (activeSection === "curiosity" && curiosityItems.length === 0) {
      setCuriosityLoading(true);
      getCuriosityQueue()
        .then((v) => setCuriosityItems(Array.isArray(v) ? v : []))
        .catch(() => setCuriosityItems([]))
        .finally(() => setCuriosityLoading(false));
    }
  }, [activeSection]);

  // MED-8: Auto-refresh metrics on tab visibility change
  useEffect(() => {
    const handleVisibility = () => {
      if (document.visibilityState === "visible") {
        refreshMetrics();
      }
    };
    document.addEventListener("visibilitychange", handleVisibility);
    return () => document.removeEventListener("visibilitychange", handleVisibility);
  }, []);

  // MED-3: Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (finetunePollingRef.current) {
        clearInterval(finetunePollingRef.current);
      }
    };
  }, []);

  // ── MED-8: Refresh metrics ──

  const refreshMetrics = useCallback(async () => {
    setMetricsRefreshing(true);
    try {
      const m = await getLearningMetrics();
      store.setMetrics(m);
    } catch {
      // silent
    } finally {
      setMetricsRefreshing(false);
    }
  }, []);

  // ── MED-2: Training data load more ──

  const handleTrainingLoadMore = useCallback(async () => {
    setTrainingLoading(true);
    try {
      const more = await getTrainingData(TRAINING_PAGE_SIZE, trainingOffset);
      const arr = Array.isArray(more) ? more : [];
      setTrainingData((prev) => [...prev, ...arr]);
      setTrainingOffset((prev) => prev + arr.length);
      setTrainingHasMore(arr.length === TRAINING_PAGE_SIZE);
    } catch {
      // silent
    } finally {
      setTrainingLoading(false);
    }
  }, [trainingOffset]);

  // ── KG loading ──

  const loadKGFacts = useCallback(async (offset: number, search: string) => {
    setKgLoading(true);
    try {
      const facts = await getKGFacts(KG_PAGE_SIZE, offset, search);
      if (offset === 0) {
        setKgFacts(facts);
      } else {
        setKgFacts((prev) => [...prev, ...facts]);
      }
      setKgHasMore(facts.length === KG_PAGE_SIZE);
      setKgOffset(offset + facts.length);
    } catch {
      if (offset === 0) setKgFacts([]);
    } finally {
      setKgLoading(false);
    }
  }, []);

  const handleKGSearch = () => {
    setKgOffset(0);
    setKgFacts([]);
    loadKGFacts(0, kgSearch);
  };

  const handleKGLoadMore = () => {
    loadKGFacts(kgOffset, kgSearch);
  };

  // ── Handlers ──

  const handleDeleteConfirm = async () => {
    if (!confirm) return;
    try {
      if (confirm.type === "bulk-lessons" && confirm.ids) {
        await bulkDeleteLessons(confirm.ids);
        confirm.ids.forEach((id) => store.removeLesson(id));
        setSelectedLessons(new Set());
        toast.success(`${confirm.ids.length} lesson${confirm.ids.length !== 1 ? "s" : ""} deleted`);
      } else if (confirm.type === "lesson") {
        await deleteLesson(confirm.id);
        store.removeLesson(confirm.id);
        setSelectedLessons((prev) => { const next = new Set(prev); next.delete(confirm.id); return next; });
        toast.success("Lesson deleted");
      } else {
        await deleteSkill(confirm.id);
        store.removeSkill(confirm.id);
        toast.success("Skill deleted");
      }
    } catch {
      toast.error(`Failed to delete ${confirm.type === "bulk-lessons" ? "lessons" : confirm.type}`);
    }
    setConfirm(null);
  };

  // MED-5: Skill toggle with loading state
  const handleToggle = async (id: number, enabled: boolean) => {
    setTogglingSkillId(id);
    try {
      await toggleSkill(id, !enabled);
      store.updateSkillEnabled(id, !enabled);
    } catch {
      toast.error("Failed to toggle skill");
    } finally {
      setTogglingSkillId(null);
    }
  };

  const handleExportTraining = async () => {
    try {
      const blob = await exportTrainingData();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "training_data_dpo.json";
      a.click();
      URL.revokeObjectURL(url);
      toast.success("Training data exported");
    } catch {
      toast.error("Failed to export training data");
    }
  };

  // ── MED-3: Fine-tune trigger with polling ──

  const handleTriggerFinetune = async () => {
    setFinetuneRunning(true);
    try {
      const result = await triggerFinetune();
      toast.success(result.message || "Fine-tuning started successfully");

      // Start polling finetune status every 10 seconds
      if (finetunePollingRef.current) clearInterval(finetunePollingRef.current);
      finetunePollingRef.current = setInterval(async () => {
        try {
          const status = await getFinetuneStatus();
          setFinetuneStatus(status);
          // Stop polling when no longer running
          if (!status || (status as FinetuneStatus & { status?: string }).status !== "running") {
            if (finetunePollingRef.current) {
              clearInterval(finetunePollingRef.current);
              finetunePollingRef.current = null;
            }
            setFinetuneRunning(false);
          }
        } catch {
          if (finetunePollingRef.current) {
            clearInterval(finetunePollingRef.current);
            finetunePollingRef.current = null;
          }
          setFinetuneRunning(false);
        }
      }, 10000);
    } catch (err) {
      toast.error(`Fine-tuning failed: ${(err as Error).message}`);
      setFinetuneRunning(false);
    }
  };

  // ── MED-7: Sorted lessons ──

  const sortedLessons = [...store.lessons].sort((a, b) => {
    let cmp = 0;
    if (sortField === "confidence") cmp = a.confidence - b.confidence;
    else if (sortField === "times_retrieved") cmp = a.times_retrieved - b.times_retrieved;
    else if (sortField === "created_at") cmp = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
    return sortDir === "desc" ? -cmp : cmp;
  });

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortField(field);
      setSortDir("desc");
    }
  };

  const SortArrow = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null;
    return sortDir === "desc" ? <ArrowDown size={12} className="inline ml-0.5" /> : <ArrowUp size={12} className="inline ml-0.5" />;
  };

  // MED-11: Bulk selection helpers
  const allSelected = store.lessons.length > 0 && selectedLessons.size === store.lessons.length;
  const someSelected = selectedLessons.size > 0;

  const toggleSelectAll = () => {
    if (allSelected) {
      setSelectedLessons(new Set());
    } else {
      setSelectedLessons(new Set(store.lessons.map((l) => l.id)));
    }
  };

  const toggleSelectLesson = (id: number) => {
    setSelectedLessons((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  if (store.loading) {
    return (
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl w-full px-4 md:px-6 py-6">
          <Skeleton lines={6} />
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl w-full px-4 md:px-6 py-6">
        <PageHeader icon={<GraduationCap size={22} />} title="Learning Dashboard" />

        {/* Metric cards — MED-8: refresh button + auto-refresh */}
        {store.metrics && (
          <div className="mb-8">
            <div className="mb-2 flex justify-end">
              <button
                onClick={refreshMetrics}
                disabled={metricsRefreshing}
                className="flex items-center gap-1.5 text-xs text-nova-text-dim hover:text-nova-text transition-colors disabled:opacity-40"
                title="Refresh metrics"
              >
                <RefreshCw size={12} className={metricsRefreshing ? "animate-spin" : ""} />
                Refresh
              </button>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard label="Lessons" value={store.metrics.total_lessons} />
              <StatCard label="Skills" value={store.metrics.total_skills} />
              <StatCard label="Corrections" value={store.metrics.total_corrections} />
              <StatCard
                label="Training Examples"
                value={store.metrics.training_examples}
                sub={
                  store.metrics.last_correction_date
                    ? `Last: ${formatDate(store.metrics.last_correction_date)}`
                    : undefined
                }
              />
            </div>
          </div>
        )}

        {/* Section tabs */}
        <Tabs
          tabs={[
            { id: "lessons", label: `Lessons (${store.lessons.length})` },
            { id: "skills", label: `Skills (${store.skills.length})`, badge: autoSkills },
            { id: "reflexions", label: "Reflexions" },
            { id: "training", label: "Training Data" },
            { id: "knowledge", label: "Knowledge Graph" },
            { id: "curiosity", label: "Curiosity" },
          ]}
          active={activeSection}
          onChange={(id) => setActiveSection(id as Section)}
        />

        {/* Lessons section — MED-7: sortable, MED-11: bulk delete */}
        {activeSection === "lessons" && (
          <section>
            {/* MED-11: Bulk delete bar */}
            {someSelected && (
              <div className="mb-3 flex items-center gap-3 rounded-lg border border-nova-border bg-nova-surface px-3 py-2">
                <span className="text-sm text-nova-text-dim">
                  {selectedLessons.size} selected
                </span>
                <Button
                  variant="danger"
                  size="sm"
                  onClick={() =>
                    setConfirm({ type: "bulk-lessons", id: 0, ids: Array.from(selectedLessons) })
                  }
                >
                  Delete selected ({selectedLessons.size})
                </Button>
              </div>
            )}

            {store.lessons.length === 0 ? (
              <EmptyState
                icon={<GraduationCap size={40} strokeWidth={1.5} />}
                title="No lessons learned yet."
              />
            ) : (
              <div className="overflow-x-auto rounded-lg border border-nova-border">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-nova-border bg-nova-surface text-left text-xs text-nova-text-dim">
                      {/* MED-11: Select all checkbox */}
                      <th className="px-3 py-2 w-8">
                        <input
                          type="checkbox"
                          checked={allSelected}
                          onChange={toggleSelectAll}
                          className="accent-nova-accent"
                        />
                      </th>
                      <th className="px-3 py-2">Topic</th>
                      <th className="px-3 py-2">Correct Answer</th>
                      <th
                        className="px-3 py-2 text-center cursor-pointer select-none hover:text-nova-text transition-colors"
                        onClick={() => handleSort("confidence")}
                      >
                        Confidence <SortArrow field="confidence" />
                      </th>
                      <th
                        className="px-3 py-2 text-center cursor-pointer select-none hover:text-nova-text transition-colors"
                        onClick={() => handleSort("times_retrieved")}
                      >
                        Retrieved <SortArrow field="times_retrieved" />
                      </th>
                      <th
                        className="px-3 py-2 text-center cursor-pointer select-none hover:text-nova-text transition-colors"
                        onClick={() => handleSort("created_at")}
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
                            onChange={() => toggleSelectLesson(l.id)}
                            className="accent-nova-accent"
                          />
                        </td>
                        <td className="max-w-[200px] truncate px-3 py-2 font-medium">{l.topic}</td>
                        <td className="max-w-[250px] truncate px-3 py-2 text-nova-text-dim">{l.correct_answer}</td>
                        <td className="px-3 py-2 text-center">{pct(l.confidence)}</td>
                        <td className="px-3 py-2 text-center">{l.times_retrieved}</td>
                        <td className="px-3 py-2 text-center text-xs text-nova-text-dim">{formatDate(l.created_at)}</td>
                        <td className="px-3 py-2 text-right">
                          <Button variant="ghost" size="sm" onClick={() => setConfirm({ type: "lesson", id: l.id })}>
                            Delete
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}

        {/* Skills section — MED-5: per-skill loading state */}
        {activeSection === "skills" && (
          <section>
            {store.skills.length === 0 ? (
              <EmptyState
                icon={<GraduationCap size={40} strokeWidth={1.5} />}
                title="No skills acquired yet."
              />
            ) : (
              <div className="overflow-x-auto rounded-lg border border-nova-border">
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
                    {store.skills.map((s) => (
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
                            onClick={() => handleToggle(s.id, s.enabled)}
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
                          <Button variant="ghost" size="sm" onClick={() => setConfirm({ type: "skill", id: s.id })}>
                            Delete
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}

        {/* Reflexions section */}
        {activeSection === "reflexions" && (
          <section>
            {reflexionsLoading ? (
              <Skeleton lines={4} />
            ) : reflexions.length === 0 ? (
              <EmptyState
                icon={<GraduationCap size={40} strokeWidth={1.5} />}
                title="No reflexions recorded yet."
                description="These are created automatically when Nova detects low-quality answers."
              />
            ) : (
              <div className="space-y-3">
                {reflexions.map((r) => (
                  <Card key={r.id}>
                    <div className="mb-1 flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span
                          className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                            r.outcome === "failure"
                              ? "bg-nova-error/20 text-nova-error"
                              : "bg-nova-success/20 text-nova-success"
                          }`}
                        >
                          {r.outcome}
                        </span>
                        <span className="text-xs text-nova-text-dim">
                          Quality: {Math.round(r.quality_score * 100)}%
                        </span>
                      </div>
                      <span className="text-xs text-nova-text-dim">{formatDate(r.created_at)}</span>
                    </div>
                    <p className="mb-1 text-sm font-medium">{r.task_summary}</p>
                    <p className="text-sm text-nova-text-dim">{r.reflection}</p>
                    {r.tools_used.length > 0 && (
                      <div className="mt-1 flex gap-1">
                        {(typeof r.tools_used === "string" ? (r.tools_used as string).split(",") : r.tools_used)
                          .filter(Boolean)
                          .map((t: string, i: number) => (
                            <span
                              key={i}
                              className="rounded bg-nova-border px-1.5 py-0.5 text-[10px] text-nova-text-dim"
                            >
                              {t.trim()}
                            </span>
                          ))}
                      </div>
                    )}
                  </Card>
                ))}
              </div>
            )}
          </section>
        )}

        {/* Training data section — MED-2: load more, MED-3: polling status */}
        {activeSection === "training" && (
          <section>
            {trainingLoading && trainingData.length === 0 ? (
              <Skeleton lines={4} />
            ) : (
              <>
                {/* HIGH-2: Training Data Stats */}
                {trainingStats && trainingStats.total_pairs > 0 && (
                  <Card className="mb-4">
                    <h3 className="mb-2 text-xs font-medium text-nova-text-dim">Training Data Stats</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                      <div>
                        <span className="text-xs text-nova-text-dim">Total Pairs</span>
                        <p className="font-medium">{trainingStats.total_pairs}</p>
                      </div>
                      <div>
                        <span className="text-xs text-nova-text-dim">Valid Pairs</span>
                        <p className="font-medium">{trainingStats.valid_pairs}</p>
                      </div>
                      <div>
                        <span className="text-xs text-nova-text-dim">Avg Chosen Length</span>
                        <p className="font-medium">{trainingStats.avg_chosen_length} chars</p>
                      </div>
                      <div>
                        <span className="text-xs text-nova-text-dim">Avg Rejected Length</span>
                        <p className="font-medium">{trainingStats.avg_rejected_length} chars</p>
                      </div>
                    </div>
                    {trainingStats.date_range && trainingStats.date_range.earliest && (
                      <div className="mt-2 text-xs text-nova-text-dim">
                        Date range: {formatDate(trainingStats.date_range.earliest)} — {trainingStats.date_range.latest ? formatDate(trainingStats.date_range.latest) : "now"}
                      </div>
                    )}
                  </Card>
                )}

                {finetuneStatus && (
                  <div className="mb-4 flex flex-wrap items-center gap-4 rounded-lg border border-nova-border p-3">
                    <div>
                      <span className="text-sm font-medium">{finetuneStatus.training_pairs} pairs</span>
                      <span className="ml-2 text-xs text-nova-text-dim">
                        (min: {finetuneStatus.min_required})
                      </span>
                    </div>
                    <span
                      className={`rounded px-2 py-0.5 text-xs font-medium ${
                        finetuneStatus.ready
                          ? "bg-nova-success/20 text-nova-success"
                          : "bg-nova-error/20 text-nova-error"
                      }`}
                    >
                      {finetuneStatus.ready ? "Ready for fine-tuning" : "Not enough data"}
                    </span>
                    {/* MED-3: Running badge */}
                    {finetuneRunning && (
                      <span className="flex items-center gap-1.5 rounded px-2 py-0.5 text-xs font-medium bg-nova-warning/20 text-nova-warning">
                        <Loader2 size={12} className="animate-spin" />
                        Running...
                      </span>
                    )}
                    <div className="ml-auto flex gap-2">
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={handleTriggerFinetune}
                        disabled={!finetuneStatus.ready || finetuneRunning}
                        loading={finetuneRunning}
                        icon={<Play size={14} />}
                      >
                        Start Fine-Tuning
                      </Button>
                      <Button
                        size="sm"
                        onClick={handleExportTraining}
                        disabled={trainingData.length === 0}
                        icon={<Download size={14} />}
                      >
                        Export DPO
                      </Button>
                    </div>
                  </div>
                )}

                {trainingData.length === 0 ? (
                  <EmptyState
                    icon={<GraduationCap size={40} strokeWidth={1.5} />}
                    title="No training pairs yet."
                    description="Correct Nova to generate them."
                  />
                ) : (
                  <>
                    <div className="space-y-2">
                      {trainingData.map((entry, i) => (
                        <Card key={i}>
                          <div className="mb-1 flex items-center justify-between">
                            <span className="font-medium text-sm">Q: {entry.query}</span>
                            {entry.created_at && (
                              <span className="text-xs text-nova-text-dim">{formatDate(entry.created_at)}</span>
                            )}
                          </div>
                          <div className="grid grid-cols-2 gap-2 text-xs">
                            <div>
                              <span className="text-nova-success">Chosen:</span>
                              <p className="text-nova-text-dim">{entry.chosen.slice(0, 150)}</p>
                            </div>
                            <div>
                              <span className="text-nova-error">Rejected:</span>
                              <p className="text-nova-text-dim">{entry.rejected.slice(0, 150)}</p>
                            </div>
                          </div>
                        </Card>
                      ))}
                    </div>

                    {/* MED-2: Load More button */}
                    {trainingHasMore && (
                      <div className="mt-4 flex justify-center">
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={handleTrainingLoadMore}
                          loading={trainingLoading}
                        >
                          Load More
                        </Button>
                      </div>
                    )}
                  </>
                )}

                {/* HIGH-1: Finetune History */}
                {finetuneHistory.length > 0 && (
                  <div className="mt-8">
                    <h3 className="mb-3 text-sm font-medium text-nova-text-dim">
                      Fine-Tune History ({finetuneHistory.length})
                    </h3>
                    <div className="overflow-x-auto rounded-lg border border-nova-border">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-nova-border bg-nova-surface text-left text-xs text-nova-text-dim">
                            <th className="px-3 py-2">Date</th>
                            <th className="px-3 py-2">Status</th>
                            <th className="px-3 py-2 text-center">Pairs</th>
                            <th className="px-3 py-2 text-center">New</th>
                            <th className="px-3 py-2">Eval Results</th>
                          </tr>
                        </thead>
                        <tbody>
                          {finetuneHistory.map((run, i) => (
                            <tr
                              key={i}
                              className="border-b border-nova-border last:border-0 hover:bg-nova-surface/50"
                            >
                              <td className="whitespace-nowrap px-3 py-2 text-nova-text-dim">
                                {run.started_at ? formatDate(run.started_at) : "—"}
                              </td>
                              <td className="px-3 py-2">
                                <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                                  run.status === "deployed" || run.status === "success"
                                    ? "bg-nova-success/20 text-nova-success"
                                    : run.status === "rejected" || run.status === "failed"
                                    ? "bg-nova-error/20 text-nova-error"
                                    : "bg-nova-border text-nova-text-dim"
                                }`}>
                                  {run.status || "unknown"}
                                </span>
                              </td>
                              <td className="px-3 py-2 text-center">{run.training_pairs}</td>
                              <td className="px-3 py-2 text-center">{run.new_pairs}</td>
                              <td className="px-3 py-2 text-xs text-nova-text-dim">
                                {run.eval ? (
                                  <span>
                                    Win rate: {run.eval.win_rate !== null ? `${Math.round(run.eval.win_rate * 100)}%` : "—"}
                                    {run.eval.candidate_is_better !== null && (
                                      <span className={`ml-1.5 rounded px-1 py-0.5 text-[9px] font-medium ${
                                        run.eval.candidate_is_better
                                          ? "bg-nova-success/20 text-nova-success"
                                          : "bg-nova-error/20 text-nova-error"
                                      }`}>
                                        {run.eval.candidate_is_better ? "better" : "worse"}
                                      </span>
                                    )}
                                  </span>
                                ) : run.reason || "—"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </>
            )}
          </section>
        )}

        {/* Knowledge Graph section — MED-10: temporal columns */}
        {activeSection === "knowledge" && (
          <section>
            <div className="mb-4 flex gap-2">
              <FormInput
                value={kgSearch}
                onChange={(e) => setKgSearch(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleKGSearch()}
                placeholder="Search knowledge graph facts..."
                icon={<Search size={14} />}
                className="flex-1"
              />
              <Button onClick={handleKGSearch} loading={kgLoading && kgOffset === 0}>
                Search
              </Button>
            </div>

            {kgLoading && kgFacts.length === 0 ? (
              <Skeleton lines={6} />
            ) : kgFacts.length === 0 ? (
              <EmptyState
                icon={<Database size={40} strokeWidth={1.5} />}
                title="No knowledge graph facts found."
                description={kgSearch ? "Try a different search term." : "Facts are extracted from monitors, conversations, and domain studies."}
              />
            ) : (
              <>
                <div className="mb-2 text-xs text-nova-text-dim">
                  Showing {kgFacts.length} fact{kgFacts.length !== 1 ? "s" : ""}
                  {kgSearch && ` matching "${kgSearch}"`}
                </div>

                <div className="overflow-x-auto rounded-lg border border-nova-border">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-nova-border bg-nova-surface text-left text-xs text-nova-text-dim">
                        <th className="px-3 py-2">Subject</th>
                        <th className="px-3 py-2">Predicate</th>
                        <th className="px-3 py-2">Object</th>
                        <th className="px-3 py-2 text-center">Confidence</th>
                        <th className="px-3 py-2">Source</th>
                        {/* MED-10: Temporal columns */}
                        <th className="px-3 py-2">Valid From</th>
                        <th className="px-3 py-2">Valid To</th>
                        <th className="px-3 py-2">Created</th>
                      </tr>
                    </thead>
                    <tbody>
                      {kgFacts.map((fact) => (
                        <tr
                          key={fact.id}
                          className="border-b border-nova-border last:border-0 hover:bg-nova-surface/50"
                        >
                          <td className="max-w-[150px] truncate px-3 py-2 font-medium">{fact.subject}</td>
                          <td className="max-w-[120px] truncate px-3 py-2 text-nova-accent">{fact.predicate}</td>
                          <td className="max-w-[200px] truncate px-3 py-2 text-nova-text-dim">{fact.object}</td>
                          <td className="px-3 py-2 text-center">
                            <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                              fact.confidence >= 0.8
                                ? "bg-nova-success/20 text-nova-success"
                                : fact.confidence >= 0.5
                                  ? "bg-nova-warning/20 text-nova-warning"
                                  : "bg-nova-error/20 text-nova-error"
                            }`}>
                              {pct(fact.confidence)}
                            </span>
                          </td>
                          <td className="max-w-[100px] truncate px-3 py-2 text-xs text-nova-text-dim">{fact.source}</td>
                          {/* MED-10: valid_from */}
                          <td className="px-3 py-2 text-xs text-nova-text-dim">
                            {fact.valid_from ? formatDate(fact.valid_from) : "—"}
                          </td>
                          {/* MED-10: valid_to with "current" badge */}
                          <td className="px-3 py-2 text-xs">
                            {fact.valid_to === null || fact.valid_to === undefined ? (
                              <span className="rounded px-1.5 py-0.5 text-[10px] font-medium bg-nova-success/20 text-nova-success">
                                current
                              </span>
                            ) : (
                              <span className="text-nova-text-dim">{formatDate(fact.valid_to)}</span>
                            )}
                          </td>
                          <td className="px-3 py-2 text-xs text-nova-text-dim">{formatDate(fact.created_at)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {kgHasMore && (
                  <div className="mt-4 flex justify-center">
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={handleKGLoadMore}
                      loading={kgLoading}
                    >
                      Load More
                    </Button>
                  </div>
                )}
              </>
            )}
          </section>
        )}

        {/* HIGH-3: Curiosity Queue section */}
        {activeSection === "curiosity" && (
          <section>
            {curiosityLoading ? (
              <Skeleton lines={4} />
            ) : curiosityItems.length === 0 ? (
              <EmptyState
                icon={<Search size={40} strokeWidth={1.5} />}
                title="No curiosity items queued."
                description="Nova generates curiosity items when it detects knowledge gaps during conversation."
              />
            ) : (
              <div className="space-y-2">
                {curiosityItems.map((item) => (
                  <Card key={item.id}>
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium">{item.question}</p>
                        <div className="mt-1 flex flex-wrap items-center gap-2">
                          <span className="rounded bg-nova-border/40 px-1.5 py-0.5 text-[10px] text-nova-text-dim">
                            {item.source}
                          </span>
                          <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                            item.status === "completed" || item.status === "researched"
                              ? "bg-nova-success/20 text-nova-success"
                              : item.status === "pending"
                              ? "bg-nova-border text-nova-text-dim"
                              : "bg-nova-warning/20 text-nova-warning"
                          }`}>
                            {item.status}
                          </span>
                          <span className="text-[10px] text-nova-text-dim">
                            Priority: {item.priority}
                          </span>
                        </div>
                      </div>
                      <span className="shrink-0 text-xs text-nova-text-dim">
                        {formatDate(item.created_at)}
                      </span>
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </section>
        )}

        {/* Confirm dialog */}
        {confirm && (
          <ConfirmDialog
            message={
              confirm.type === "bulk-lessons"
                ? `Delete ${confirm.ids?.length} selected lesson${(confirm.ids?.length ?? 0) !== 1 ? "s" : ""}? This cannot be undone.`
                : `Delete this ${confirm.type}? This cannot be undone.`
            }
            onConfirm={handleDeleteConfirm}
            onCancel={() => setConfirm(null)}
          />
        )}
      </div>
    </div>
  );
}
