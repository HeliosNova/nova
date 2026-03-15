import { useEffect, useState } from "react";
import { Activity, Plus, Pencil, ThumbsUp, ThumbsDown, ChevronDown, ChevronUp, Trash2, MessageSquare } from "lucide-react";
import { toast } from "sonner";
import {
  getMonitors,
  createMonitor,
  updateMonitor,
  deleteMonitor,
  triggerMonitor,
  getRecentMonitorResults,
  getMonitorDetail,
  rateMonitorResult,
  getHeartbeatInstructions,
  createHeartbeatInstruction,
  deleteHeartbeatInstruction,
} from "../lib/api";
import { formatDate, formatSeconds } from "../lib/utils";
import type { MonitorInfo, MonitorResult, MonitorDetail } from "../lib/types";
import {
  PageHeader,
  Card,
  Button,
  FormInput,
  FormSelect,
  Modal,
  ConfirmDialog,
  EmptyState,
  Skeleton,
} from "../components/ui";

function configTarget(cfg: Record<string, unknown>): string {
  return String(cfg?.url || cfg?.query || cfg?.command || cfg?.target || "—");
}

export default function MonitorsPage() {
  const [monitors, setMonitors] = useState<MonitorInfo[]>([]);
  const [results, setResults] = useState<MonitorResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [confirmId, setConfirmId] = useState<number | null>(null);
  const [triggering, setTriggering] = useState<number | null>(null);

  // Create/Edit form
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [formName, setFormName] = useState("");
  const [formType, setFormType] = useState("url");
  const [formTarget, setFormTarget] = useState("");
  const [formSchedule, setFormSchedule] = useState(3600);
  const [saving, setSaving] = useState(false);

  // MED-6: Form validation
  const [formTouched, setFormTouched] = useState<Record<string, boolean>>({});

  // MED-12: Results pagination
  const RESULTS_PAGE_SIZE = 20;
  const [resultsVisible, setResultsVisible] = useState(RESULTS_PAGE_SIZE);

  // Detail modal
  const [detailMonitor, setDetailMonitor] = useState<MonitorDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const safeSetMonitors = (v: unknown) => setMonitors(Array.isArray(v) ? v : []);
  const safeSetResults = (v: unknown) => setResults(Array.isArray(v) ? v : []);

  const refresh = () => {
    setLoading(true);
    Promise.all([
      getMonitors().then(safeSetMonitors).catch(() => setMonitors([])),
      getRecentMonitorResults().then(safeSetResults).catch(() => setResults([])),
    ]).finally(() => setLoading(false));
  };

  useEffect(() => {
    refresh();
  }, []);

  const resetForm = () => {
    setFormName("");
    setFormType("url");
    setFormTarget("");
    setFormSchedule(3600);
    setEditingId(null);
    setShowForm(false);
    setFormTouched({});
  };

  // MED-6: Validation helpers
  const nameError = formTouched.name
    ? !formName.trim()
      ? "Name is required"
      : formName.trim().length < 2
        ? "Name must be at least 2 characters"
        : undefined
    : undefined;

  const scheduleError = formTouched.schedule
    ? !formSchedule || formSchedule <= 0
      ? "Schedule must be a positive number"
      : undefined
    : undefined;

  const scheduleHint = formSchedule > 0 ? `= ${formatSeconds(formSchedule)}` : undefined;

  const formValid = formName.trim().length >= 2 && formTarget.trim().length > 0 && formSchedule > 0;

  const handleDelete = async () => {
    if (confirmId === null) return;
    try {
      await deleteMonitor(confirmId);
      setConfirmId(null);
      toast.success("Monitor deleted");
      refresh();
    } catch {
      toast.error("Failed to delete monitor");
      setConfirmId(null);
    }
  };

  const handleTrigger = async (id: number) => {
    setTriggering(id);
    try {
      await triggerMonitor(id);
      toast.success("Monitor triggered");
      refresh();
    } catch {
      toast.error("Failed to trigger monitor");
    } finally {
      setTriggering(null);
    }
  };

  const handleRate = async (resultId: number, rating: -1 | 1) => {
    try {
      await rateMonitorResult(resultId, rating);
      toast.success(rating === 1 ? "Rated as good" : "Rated as bad");
    } catch {
      toast.error("Failed to rate result");
    }
  };

  const startEdit = (m: MonitorInfo) => {
    setEditingId(m.id);
    setFormName(m.name);
    setFormType(m.check_type);
    setFormTarget(configTarget(m.check_config));
    setFormSchedule(m.schedule_seconds);
    setShowForm(true);
  };

  const handleSave = async () => {
    setFormTouched({ name: true, schedule: true });
    if (!formValid) return;
    setSaving(true);
    try {
      const config: Record<string, unknown> =
        formType === "url" ? { url: formTarget } :
        formType === "search" ? { query: formTarget } :
        formType === "query" ? { query: formTarget } :
        { command: formTarget };

      if (editingId !== null) {
        await updateMonitor(editingId, {
          name: formName,
          check_type: formType,
          check_config: config,
          schedule_seconds: formSchedule,
        });
        toast.success("Monitor updated");
      } else {
        await createMonitor({
          name: formName,
          check_type: formType,
          check_config: config,
          schedule_seconds: formSchedule,
        });
        toast.success("Monitor created");
      }
      resetForm();
      refresh();
    } catch {
      toast.error(editingId !== null ? "Failed to update monitor" : "Failed to create monitor");
    } finally {
      setSaving(false);
    }
  };

  const openDetail = async (id: number) => {
    setDetailLoading(true);
    try {
      const detail = await getMonitorDetail(id);
      setDetailMonitor(detail);
    } catch {
      toast.error("Failed to load monitor details");
    } finally {
      setDetailLoading(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl w-full px-4 md:px-6 py-6">
        <PageHeader
          icon={<Activity size={22} />}
          title="Monitors"
          actions={
            <Button
              onClick={() => {
                if (showForm) {
                  resetForm();
                } else {
                  setEditingId(null);
                  setShowForm(true);
                }
              }}
              icon={showForm ? undefined : <Plus size={16} />}
              variant={showForm ? "secondary" : "primary"}
            >
              {showForm ? "Cancel" : "New Monitor"}
            </Button>
          }
        />

        {/* Create/Edit form */}
        {showForm && (
          <Card className="mb-6">
            <h3 className="mb-3 text-sm font-medium text-nova-text-dim">
              {editingId !== null ? "Edit Monitor" : "New Monitor"}
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <FormInput
                label="Name"
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
                onBlur={() => setFormTouched((t) => ({ ...t, name: true }))}
                placeholder="Monitor name"
                error={nameError}
              />
              <FormSelect
                label="Type"
                value={formType}
                onChange={(e) => setFormType(e.target.value)}
                options={[
                  { value: "url", label: "URL" },
                  { value: "search", label: "Search" },
                  { value: "command", label: "Command" },
                  { value: "query", label: "Query" },
                ]}
              />
              <FormInput
                label="Target"
                value={formTarget}
                onChange={(e) => setFormTarget(e.target.value)}
                placeholder="Target URL, query, or command"
              />
              <div className="space-y-1">
                <FormInput
                  label="Schedule (seconds)"
                  type="number"
                  value={String(formSchedule)}
                  onChange={(e) => setFormSchedule(Number(e.target.value))}
                  onBlur={() => setFormTouched((t) => ({ ...t, schedule: true }))}
                  min={1}
                  error={scheduleError}
                />
                {!scheduleError && scheduleHint && (
                  <p className="text-xs text-nova-text-dim">{scheduleHint}</p>
                )}
              </div>
            </div>
            <div className="mt-3 flex gap-2">
              <Button
                onClick={handleSave}
                loading={saving}
                disabled={!formValid}
              >
                {editingId !== null ? "Update Monitor" : "Create Monitor"}
              </Button>
              {editingId !== null && (
                <Button variant="secondary" onClick={resetForm}>
                  Cancel Edit
                </Button>
              )}
            </div>
          </Card>
        )}

        {/* Monitor list */}
        <section className="mb-8">
          <h2 className="mb-3 text-sm font-medium text-nova-text-dim">
            Active Monitors ({monitors.length})
          </h2>
          {loading ? (
            <Skeleton lines={3} />
          ) : monitors.length === 0 ? (
            <EmptyState
              icon={<Activity size={40} strokeWidth={1.5} />}
              title="No monitors configured."
            />
          ) : (
            <div className="space-y-2">
              {monitors.map((m) => {
                const target = configTarget(m.check_config);
                const lastStatus = m.last_result;
                return (
                  <div
                    key={m.id}
                    className="flex items-center justify-between rounded-lg border border-nova-border px-4 py-3"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span
                          className={`inline-block h-2 w-2 rounded-full ${
                            lastStatus === "ok" || lastStatus === "success"
                              ? "bg-nova-success"
                              : lastStatus === "error" || lastStatus === "fail"
                              ? "bg-nova-error"
                              : "bg-nova-text-dim"
                          }`}
                        />
                        <button
                          onClick={() => openDetail(m.id)}
                          className="font-medium text-sm text-nova-text hover:text-nova-accent transition-colors"
                        >
                          {m.name}
                        </button>
                        <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                          m.enabled ? "bg-nova-success/20 text-nova-success" : "bg-nova-error/20 text-nova-error"
                        }`}>
                          {m.enabled ? "ON" : "OFF"}
                        </span>
                      </div>
                      <div className="mt-0.5 text-xs text-nova-text-dim">
                        {m.check_type} &middot; {target.slice(0, 60)} &middot; every {Math.round(m.schedule_seconds / 60)}m
                        {m.last_check_at && <> &middot; last: {formatDate(m.last_check_at)}</>}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => startEdit(m)}
                        icon={<Pencil size={13} />}
                      >
                        Edit
                      </Button>
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => handleTrigger(m.id)}
                        loading={triggering === m.id}
                      >
                        Trigger
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setConfirmId(m.id)}
                      >
                        Delete
                      </Button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        {/* Recent results — MED-12: paginated */}
        <section>
          <h2 className="mb-3 text-sm font-medium text-nova-text-dim">
            Recent Results ({results.length})
          </h2>
          {results.length === 0 ? (
            <p className="text-sm text-nova-text-dim">No results yet.</p>
          ) : (
            <>
              <div className="overflow-x-auto rounded-lg border border-nova-border">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-nova-border bg-nova-surface text-left text-xs text-nova-text-dim">
                      <th className="px-3 py-2">Monitor</th>
                      <th className="px-3 py-2">Status</th>
                      <th className="px-3 py-2">Message</th>
                      <th className="px-3 py-2">Time</th>
                      <th className="px-3 py-2">Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.slice(0, resultsVisible).map((r) => (
                      <tr
                        key={r.id}
                        className="border-b border-nova-border last:border-0 hover:bg-nova-surface/50"
                      >
                        <td className="px-3 py-2 font-medium">#{r.monitor_id}</td>
                        <td className="px-3 py-2">
                          <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                            r.status === "ok" || r.status === "success" || r.status === "changed"
                              ? "bg-nova-success/20 text-nova-success"
                              : r.status === "error" || r.status === "fail"
                              ? "bg-nova-error/20 text-nova-error"
                              : "bg-nova-border text-nova-text-dim"
                          }`}>
                            {r.status}
                          </span>
                        </td>
                        <td className="max-w-[250px] truncate px-3 py-2 text-nova-text-dim">{r.message || r.value || "—"}</td>
                        <td className="px-3 py-2 text-nova-text-dim">{formatDate(r.created_at)}</td>
                        <td className="px-3 py-2">
                          <div className="flex gap-1">
                            <button
                              onClick={() => handleRate(r.id, 1)}
                              className="rounded p-0.5 text-nova-text-dim hover:text-nova-success hover:bg-nova-success/10 transition-colors"
                              title="Good result"
                            >
                              <ThumbsUp size={12} />
                            </button>
                            <button
                              onClick={() => handleRate(r.id, -1)}
                              className="rounded p-0.5 text-nova-text-dim hover:text-nova-error hover:bg-nova-error/10 transition-colors"
                              title="Bad result"
                            >
                              <ThumbsDown size={12} />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {results.length > resultsVisible && (
                <div className="mt-4 flex justify-center">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => setResultsVisible((v) => v + RESULTS_PAGE_SIZE)}
                  >
                    Load More
                  </Button>
                </div>
              )}
            </>
          )}
        </section>

        {/* Heartbeat Instructions */}
        <HeartbeatInstructions />

        {/* Monitor Detail Modal */}
        <Modal
          open={detailMonitor !== null}
          onClose={() => setDetailMonitor(null)}
          title={detailMonitor?.name || "Monitor Details"}
          size="lg"
        >
          {detailLoading ? (
            <Skeleton lines={4} />
          ) : detailMonitor && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-xs text-nova-text-dim">Type</span>
                  <p className="font-medium">{detailMonitor.check_type}</p>
                </div>
                <div>
                  <span className="text-xs text-nova-text-dim">Schedule</span>
                  <p className="font-medium">Every {Math.round(detailMonitor.schedule_seconds / 60)}m</p>
                </div>
                <div>
                  <span className="text-xs text-nova-text-dim">Target</span>
                  <p className="font-medium break-all">{configTarget(detailMonitor.check_config)}</p>
                </div>
                <div>
                  <span className="text-xs text-nova-text-dim">Status</span>
                  <p>
                    <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                      detailMonitor.enabled ? "bg-nova-success/20 text-nova-success" : "bg-nova-error/20 text-nova-error"
                    }`}>
                      {detailMonitor.enabled ? "Enabled" : "Disabled"}
                    </span>
                  </p>
                </div>
                <div>
                  <span className="text-xs text-nova-text-dim">Notify</span>
                  <p className="font-medium">{detailMonitor.notify_condition}</p>
                </div>
                <div>
                  <span className="text-xs text-nova-text-dim">Cooldown</span>
                  <p className="font-medium">{detailMonitor.cooldown_minutes}m</p>
                </div>
                {detailMonitor.last_check_at && (
                  <div>
                    <span className="text-xs text-nova-text-dim">Last Check</span>
                    <p className="font-medium">{formatDate(detailMonitor.last_check_at)}</p>
                  </div>
                )}
                <div>
                  <span className="text-xs text-nova-text-dim">Created</span>
                  <p className="font-medium">{formatDate(detailMonitor.created_at)}</p>
                </div>
              </div>

              {/* Result history */}
              {detailMonitor.results && detailMonitor.results.length > 0 && (
                <div>
                  <h4 className="mb-2 text-xs font-medium text-nova-text-dim">
                    Result History ({detailMonitor.results.length})
                  </h4>
                  <div className="max-h-60 overflow-y-auto rounded-lg border border-nova-border">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-nova-border bg-nova-surface text-left text-nova-text-dim">
                          <th className="px-2 py-1.5">Status</th>
                          <th className="px-2 py-1.5">Message</th>
                          <th className="px-2 py-1.5">Time</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detailMonitor.results.map((r) => (
                          <tr key={r.id} className="border-b border-nova-border last:border-0">
                            <td className="px-2 py-1.5">
                              <span className={`rounded px-1 py-0.5 text-[9px] font-medium ${
                                r.status === "ok" || r.status === "success" || r.status === "changed"
                                  ? "bg-nova-success/20 text-nova-success"
                                  : r.status === "error" || r.status === "fail"
                                  ? "bg-nova-error/20 text-nova-error"
                                  : "bg-nova-border text-nova-text-dim"
                              }`}>
                                {r.status}
                              </span>
                            </td>
                            <td className="max-w-[200px] truncate px-2 py-1.5 text-nova-text-dim">
                              {r.message || r.value || "—"}
                            </td>
                            <td className="px-2 py-1.5 text-nova-text-dim whitespace-nowrap">
                              {formatDate(r.created_at)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </Modal>

        {confirmId !== null && (
          <ConfirmDialog
            message="Delete this monitor? This cannot be undone."
            onConfirm={handleDelete}
            onCancel={() => setConfirmId(null)}
          />
        )}
      </div>
    </div>
  );
}

// ── Heartbeat Instructions section ──

function HeartbeatInstructions() {
  const [open, setOpen] = useState(false);
  const [instructions, setInstructions] = useState<{ id: number; instruction: string; schedule_seconds: number; enabled: boolean; last_run_at: string | null; created_at: string }[]>([]);
  const [loading, setLoading] = useState(false);
  const [newInst, setNewInst] = useState("");
  const [newSchedule, setNewSchedule] = useState(3600);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (open && instructions.length === 0) {
      setLoading(true);
      getHeartbeatInstructions()
        .then((v) => setInstructions(Array.isArray(v) ? v as typeof instructions : []))
        .catch(() => {})
        .finally(() => setLoading(false));
    }
  }, [open]);

  const handleCreate = async () => {
    if (!newInst.trim()) return;
    setSaving(true);
    try {
      await createHeartbeatInstruction({ instruction: newInst.trim(), schedule_seconds: newSchedule });
      toast.success("Instruction created");
      setNewInst("");
      const fresh = await getHeartbeatInstructions();
      setInstructions(Array.isArray(fresh) ? fresh as typeof instructions : []);
    } catch { toast.error("Failed to create instruction"); }
    finally { setSaving(false); }
  };

  const handleDeleteInst = async (id: number) => {
    try {
      await deleteHeartbeatInstruction(id);
      setInstructions((prev) => prev.filter((i) => i.id !== id));
      toast.success("Instruction deleted");
    } catch { toast.error("Failed to delete"); }
  };

  return (
    <section className="mb-8">
      <button
        onClick={() => setOpen(!open)}
        className="mb-3 flex items-center gap-2 text-sm font-medium text-nova-text-dim hover:text-nova-text transition-colors"
      >
        <MessageSquare size={14} />
        Heartbeat Instructions ({instructions.length})
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>
      {open && (
        <div className="space-y-3">
          {loading ? (
            <Skeleton lines={2} />
          ) : (
            <>
              {instructions.map((inst) => (
                <div key={inst.id} className="flex items-center justify-between rounded-lg border border-nova-border px-3 py-2 text-sm">
                  <div className="min-w-0 flex-1">
                    <p className="text-nova-text">{inst.instruction}</p>
                    <p className="text-xs text-nova-text-dim">
                      Every {Math.round(inst.schedule_seconds / 60)}m
                      {inst.last_run_at && <> &middot; last: {formatDate(inst.last_run_at)}</>}
                    </p>
                  </div>
                  <button onClick={() => handleDeleteInst(inst.id)} className="ml-2 text-nova-text-dim hover:text-nova-error transition-colors">
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
              <div className="flex gap-2">
                <FormInput
                  placeholder="New instruction..."
                  value={newInst}
                  onChange={(e) => setNewInst(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleCreate()}
                />
                <FormInput
                  type="number"
                  value={String(newSchedule)}
                  onChange={(e) => setNewSchedule(Number(e.target.value))}
                  min={60}
                  className="w-24"
                  placeholder="sec"
                />
                <Button onClick={handleCreate} loading={saving} disabled={!newInst.trim()}>
                  Add
                </Button>
              </div>
            </>
          )}
        </div>
      )}
    </section>
  );
}
