import { useState, useEffect, useCallback } from "react";
import {
  Settings, Moon, Sun, Monitor, Key, Download, Upload,
  Server, ToggleLeft, ToggleRight, Clock, Shield, Wrench, Trash2,
  AlertTriangle, CheckCircle, Loader2,
} from "lucide-react";
import { toast } from "sonner";
import { useSettingsStore } from "../lib/store";
import {
  getHealth,
  getStatus,
  getFacts,
  createFact,
  deleteFact,
  exportData,
  importData,
  getIntegrations,
  getAccessTier,
  getConfigSummary,
  getFullConfig,
  updateConfig,
  getCustomTools,
  deleteCustomTool,
} from "../lib/api";
import type {
  StatusResponse, UserFact, IntegrationInfo, AccessTierInfo,
  FullConfig, CustomToolInfo,
} from "../lib/types";
import {
  PageHeader,
  Card,
  Button,
  FormInput,
  FormSelect,
  StatCard,
  Skeleton,
  EmptyState,
  ConfirmDialog,
} from "../components/ui";

// ── Toggle Switch component ──

function Toggle({ checked, onChange, label, loading }: { checked: boolean; onChange: (v: boolean) => void; label: string; loading?: boolean }) {
  return (
    <button
      onClick={() => !loading && onChange(!checked)}
      className="flex w-full items-center justify-between rounded-lg border border-nova-border bg-nova-bg px-3 py-2.5 text-sm transition-colors hover:bg-nova-surface/50"
      disabled={loading}
    >
      <span className="text-nova-text">{label}</span>
      <span className="flex items-center gap-2">
        {loading && <Loader2 size={12} className="animate-spin text-nova-text-dim" />}
        {checked ? (
          <ToggleRight size={22} className="text-nova-success" />
        ) : (
          <ToggleLeft size={22} className="text-nova-text-dim" />
        )}
      </span>
    </button>
  );
}

// ── Section Header component ──

function SectionHeader({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <h2 className="mb-3 flex items-center gap-2 text-sm font-medium text-nova-text-dim">
      {icon}
      {title}
    </h2>
  );
}

// ── Access Tier descriptions ──

const TIER_DESCRIPTIONS: Record<string, string> = {
  sandboxed: "Most restrictive. Shell blocks system + interpreter commands. File ops only /data. Code blocks os/subprocess/socket.",
  standard: "Shell blocks system commands. File allows /data, /tmp, /home/nova. Code allows pathlib/os.path.",
  full: "Only container-escape commands blocked. Minimal code restrictions.",
  none: "No restrictions at all. All commands, imports, and paths allowed.",
};

const TIER_COLORS: Record<string, string> = {
  sandboxed: "bg-nova-success/20 text-nova-success",
  standard: "bg-nova-warning/20 text-nova-warning",
  full: "bg-nova-error/20 text-nova-error",
  none: "bg-nova-error/30 text-nova-error",
};

// ── Feature Toggle groups ──

const TOGGLE_GROUPS = {
  Core: [
    "ENABLE_EXTENDED_THINKING",
    "ENABLE_PLANNING",
    "ENABLE_CRITIQUE",
    "ENABLE_MODEL_ROUTING",
  ],
  Tools: [
    "ENABLE_SHELL_EXEC",
    "ENABLE_CUSTOM_TOOLS",
    "ENABLE_MCP",
    "ENABLE_DELEGATION",
  ],
  Autonomy: [
    "ENABLE_HEARTBEAT",
    "ENABLE_PROACTIVE",
    "ENABLE_CURIOSITY",
    "ENABLE_AUTO_SKILL_CREATION",
  ],
  Security: [
    "ENABLE_INJECTION_DETECTION",
    "REQUIRE_SIGNED_SKILLS",
  ],
  Optional: [
    "ENABLE_VOICE",
    "ENABLE_DESKTOP_AUTOMATION",
    "ENABLE_WEBHOOKS",
    "ENABLE_EMAIL_SEND",
  ],
};

const TIMEOUT_FIELDS = [
  { key: "GENERATION_TIMEOUT", label: "Generation Timeout (s)" },
  { key: "TOOL_TIMEOUT", label: "Tool Timeout (s)" },
  { key: "CODE_EXEC_TIMEOUT", label: "Code Exec Timeout (s)" },
  { key: "SHELL_EXEC_TIMEOUT", label: "Shell Exec Timeout (s)" },
  { key: "BROWSER_TIMEOUT", label: "Browser Timeout (s)" },
  { key: "MAX_TOOL_ROUNDS", label: "Max Tool Rounds" },
  { key: "MAX_CONTEXT_TOKENS", label: "Max Context Tokens" },
  { key: "MAX_HISTORY_MESSAGES", label: "Max History Messages" },
];

function formatToggleLabel(key: string): string {
  return key
    .replace(/^ENABLE_/, "")
    .replace(/^REQUIRE_/, "Require ")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function SettingsPage() {
  const { apiKey, setApiKey, health, setHealth, theme, setTheme } = useSettingsStore();
  const [keyInput, setKeyInput] = useState(apiKey);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [facts, setFacts] = useState<UserFact[]>([]);
  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");
  const [integrations, setIntegrations] = useState<IntegrationInfo[] | null>(null);
  const [accessTier, setAccessTier] = useState<AccessTierInfo | null>(null);

  // Full config state
  const [fullConfig, setFullConfig] = useState<FullConfig | null>(null);
  const [configLoading, setConfigLoading] = useState(true);

  // LLM Provider form state
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [ollamaUrl, setOllamaUrl] = useState("");
  const [openaiKey, setOpenaiKey] = useState("");
  const [anthropicKey, setAnthropicKey] = useState("");
  const [googleKey, setGoogleKey] = useState("");
  const [visionModel, setVisionModel] = useState("");
  const [fastModel, setFastModel] = useState("");
  const [heavyModel, setHeavyModel] = useState("");
  const [providerSaving, setProviderSaving] = useState(false);
  const [providerWarnings, setProviderWarnings] = useState<string[]>([]);

  // Toggles loading state (per key)
  const [toggleLoading, setToggleLoading] = useState<Record<string, boolean>>({});

  // Timeouts form state
  const [timeoutValues, setTimeoutValues] = useState<Record<string, string>>({});
  const [timeoutSaving, setTimeoutSaving] = useState(false);

  // Access tier form state
  const [selectedTier, setSelectedTier] = useState("");
  const [tierSaving, setTierSaving] = useState(false);

  // Custom tools state
  const [customTools, setCustomTools] = useState<CustomToolInfo[]>([]);
  const [customToolsLoading, setCustomToolsLoading] = useState(false);
  const [deletingTool, setDeletingTool] = useState<string | null>(null);

  // Load all data on mount
  useEffect(() => {
    getHealth().then(setHealth).catch(() => {});
    getStatus().then(setStatus).catch(() => {});
    getFacts().then((v) => setFacts(Array.isArray(v) ? v : [])).catch(() => setFacts([]));
    getIntegrations().then((v) => setIntegrations(Array.isArray(v) ? v : [])).catch(() => setIntegrations([]));
    getAccessTier().then(setAccessTier).catch(() => {});
    loadFullConfig();
    loadCustomTools();
  }, [apiKey, setHealth]);

  const loadFullConfig = useCallback(async () => {
    setConfigLoading(true);
    try {
      const config = await getFullConfig();
      setFullConfig(config);
      // Populate LLM provider form
      setProvider(String(config.LLM_PROVIDER || "ollama"));
      setModel(String(config.LLM_MODEL || ""));
      setOllamaUrl(String(config.OLLAMA_BASE_URL || "http://localhost:11434"));
      setOpenaiKey(String(config.OPENAI_API_KEY || ""));
      setAnthropicKey(String(config.ANTHROPIC_API_KEY || ""));
      setGoogleKey(String(config.GOOGLE_API_KEY || ""));
      setVisionModel(String(config.VISION_MODEL || ""));
      setFastModel(String(config.FAST_MODEL || ""));
      setHeavyModel(String(config.HEAVY_MODEL || ""));
      setSelectedTier(String(config.SYSTEM_ACCESS_LEVEL || "sandboxed"));
      // Populate timeout values
      const timeouts: Record<string, string> = {};
      for (const field of TIMEOUT_FIELDS) {
        timeouts[field.key] = String(config[field.key] ?? "");
      }
      setTimeoutValues(timeouts);
    } catch {
      toast.error("Failed to load full config");
    } finally {
      setConfigLoading(false);
    }
  }, []);

  const loadCustomTools = useCallback(async () => {
    setCustomToolsLoading(true);
    try {
      const tools = await getCustomTools();
      setCustomTools(Array.isArray(tools) ? tools : []);
    } catch {
      setCustomTools([]);
    } finally {
      setCustomToolsLoading(false);
    }
  }, []);

  // ── LLM Provider save ──
  const handleSaveProvider = async () => {
    setProviderSaving(true);
    setProviderWarnings([]);
    try {
      const updates: Record<string, unknown> = {
        LLM_PROVIDER: provider,
        LLM_MODEL: model,
      };
      if (provider === "ollama") {
        updates.OLLAMA_BASE_URL = ollamaUrl;
      }
      if (provider === "openai" && openaiKey) {
        updates.OPENAI_API_KEY = openaiKey;
      }
      if (provider === "anthropic" && anthropicKey) {
        updates.ANTHROPIC_API_KEY = anthropicKey;
      }
      if (provider === "google" && googleKey) {
        updates.GOOGLE_API_KEY = googleKey;
      }
      if (visionModel) updates.VISION_MODEL = visionModel;
      if (fastModel) updates.FAST_MODEL = fastModel;
      if (heavyModel) updates.HEAVY_MODEL = heavyModel;

      const result = await updateConfig(updates);
      if (result.warnings && result.warnings.length > 0) {
        setProviderWarnings(result.warnings);
      }
      if (result.restart_required) {
        toast.info("Restart required for changes to take effect", { duration: 5000 });
      } else {
        toast.success("Provider reinitialized");
      }
      getHealth().then(setHealth).catch(() => {});
    } catch (err) {
      toast.error(`Save failed: ${(err as Error).message}`);
    } finally {
      setProviderSaving(false);
    }
  };

  // ── Toggle handler ──
  const handleToggle = async (key: string, newValue: boolean) => {
    setToggleLoading((prev) => ({ ...prev, [key]: true }));
    try {
      const result = await updateConfig({ [key]: newValue });
      setFullConfig((prev) => prev ? { ...prev, [key]: newValue } : prev);
      if (result.warnings && result.warnings.length > 0) {
        result.warnings.forEach((w) => toast.warning(w));
      }
      if (result.restart_required) {
        toast.info("Restart required", { duration: 3000 });
      }
    } catch (err) {
      toast.error(`Toggle failed: ${(err as Error).message}`);
    } finally {
      setToggleLoading((prev) => ({ ...prev, [key]: false }));
    }
  };

  // ── Timeout save ──
  const handleSaveTimeouts = async () => {
    setTimeoutSaving(true);
    try {
      const updates: Record<string, unknown> = {};
      for (const field of TIMEOUT_FIELDS) {
        const val = timeoutValues[field.key];
        if (val !== "" && !isNaN(Number(val))) {
          updates[field.key] = Number(val);
        }
      }
      const result = await updateConfig(updates);
      if (result.warnings && result.warnings.length > 0) {
        result.warnings.forEach((w) => toast.warning(w));
      }
      toast.success("Timeouts & limits saved");
    } catch (err) {
      toast.error(`Save failed: ${(err as Error).message}`);
    } finally {
      setTimeoutSaving(false);
    }
  };

  // ── Access tier save ──
  const handleSaveTier = async () => {
    setTierSaving(true);
    try {
      const result = await updateConfig({ SYSTEM_ACCESS_LEVEL: selectedTier });
      if (result.restart_required) {
        toast.info("Restart required for tier change");
      } else {
        toast.success("Access tier updated");
      }
      getAccessTier().then(setAccessTier).catch(() => {});
    } catch (err) {
      toast.error(`Save failed: ${(err as Error).message}`);
    } finally {
      setTierSaving(false);
    }
  };

  // ── Custom tool delete ──
  const handleDeleteTool = async () => {
    if (!deletingTool) return;
    try {
      await deleteCustomTool(deletingTool);
      toast.success(`Tool "${deletingTool}" deleted`);
      setDeletingTool(null);
      loadCustomTools();
    } catch (err) {
      toast.error(`Delete failed: ${(err as Error).message}`);
      setDeletingTool(null);
    }
  };

  // ── Existing handlers (unchanged) ──

  const saveApiKey = () => {
    setApiKey(keyInput);
    toast.success("API key saved");
  };

  const handleCreateFact = async () => {
    if (!newKey.trim() || !newValue.trim()) return;
    await createFact({ key: newKey.trim(), value: newValue.trim() });
    setNewKey("");
    setNewValue("");
    const f = await getFacts(); setFacts(Array.isArray(f) ? f : []);
    toast.success("Fact added");
  };

  const handleDeleteFact = async (key: string) => {
    await deleteFact(key);
    const f = await getFacts(); setFacts(Array.isArray(f) ? f : []);
    toast.success("Fact deleted");
  };

  const handleExport = async () => {
    try {
      const blob = await exportData();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `nova-export-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("Export downloaded");
    } catch (err) {
      toast.error(`Export failed: ${(err as Error).message}`);
    }
  };

  const handleImport = async () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        const result = await importData(file);
        toast.success(`Imported: ${JSON.stringify(result.stats)}`);
        getStatus().then(setStatus).catch(() => {});
      } catch (err) {
        toast.error(`Import failed: ${(err as Error).message}`);
      }
    };
    input.click();
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-2xl w-full space-y-8 px-4 md:px-6 py-6">
        <PageHeader icon={<Settings size={22} />} title="Settings" />

        {/* ═══ Section 1: LLM Provider ═══ */}
        <Card>
          <SectionHeader icon={<Server size={16} />} title="LLM Provider" />
          {configLoading ? (
            <Skeleton lines={4} />
          ) : (
            <div className="space-y-3">
              <FormSelect
                label="Provider"
                value={provider}
                onChange={(e) => setProvider(e.target.value)}
                options={[
                  { value: "ollama", label: "Ollama (Local)" },
                  { value: "openai", label: "OpenAI" },
                  { value: "anthropic", label: "Anthropic" },
                  { value: "google", label: "Google" },
                ]}
              />
              <FormInput
                label="Model"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder="e.g. qwen3.5:27b, gpt-4o, claude-sonnet"
              />

              {provider === "ollama" && (
                <FormInput
                  label="Ollama URL"
                  value={ollamaUrl}
                  onChange={(e) => setOllamaUrl(e.target.value)}
                  placeholder="http://localhost:11434"
                />
              )}

              {provider === "openai" && (
                <FormInput
                  label="OpenAI API Key"
                  type="password"
                  value={openaiKey}
                  onChange={(e) => setOpenaiKey(e.target.value)}
                  placeholder="sk-..."
                  icon={<Key size={14} />}
                />
              )}

              {provider === "anthropic" && (
                <FormInput
                  label="Anthropic API Key"
                  type="password"
                  value={anthropicKey}
                  onChange={(e) => setAnthropicKey(e.target.value)}
                  placeholder="sk-ant-..."
                  icon={<Key size={14} />}
                />
              )}

              {provider === "google" && (
                <FormInput
                  label="Google API Key"
                  type="password"
                  value={googleKey}
                  onChange={(e) => setGoogleKey(e.target.value)}
                  placeholder="AIza..."
                  icon={<Key size={14} />}
                />
              )}

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <FormInput
                  label="Vision Model"
                  value={visionModel}
                  onChange={(e) => setVisionModel(e.target.value)}
                  placeholder="Optional"
                />
                <FormInput
                  label="Fast Model"
                  value={fastModel}
                  onChange={(e) => setFastModel(e.target.value)}
                  placeholder="Optional"
                />
                <FormInput
                  label="Heavy Model"
                  value={heavyModel}
                  onChange={(e) => setHeavyModel(e.target.value)}
                  placeholder="Optional"
                />
              </div>

              {providerWarnings.length > 0 && (
                <div className="space-y-1">
                  {providerWarnings.map((w, i) => (
                    <div key={i} className="flex items-center gap-2 rounded border border-nova-warning/30 bg-nova-warning/10 px-3 py-2 text-xs text-nova-warning">
                      <AlertTriangle size={14} />
                      {w}
                    </div>
                  ))}
                </div>
              )}

              <Button onClick={handleSaveProvider} loading={providerSaving}>
                Save Provider Settings
              </Button>
            </div>
          )}
        </Card>

        {/* ═══ Section 2: Feature Toggles ═══ */}
        <Card>
          <SectionHeader icon={<ToggleRight size={16} />} title="Feature Toggles" />
          {configLoading || !fullConfig ? (
            <Skeleton lines={6} />
          ) : (
            <div className="space-y-4">
              {Object.entries(TOGGLE_GROUPS).map(([group, keys]) => (
                <div key={group}>
                  <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-nova-text-dim/70">{group}</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-1.5">
                    {keys.map((key) => {
                      const val = fullConfig[key];
                      // Only render if the config key exists
                      if (val === undefined) return null;
                      return (
                        <Toggle
                          key={key}
                          label={formatToggleLabel(key)}
                          checked={Boolean(val)}
                          loading={toggleLoading[key]}
                          onChange={(v) => handleToggle(key, v)}
                        />
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* ═══ Section 3: Timeouts & Limits ═══ */}
        <Card>
          <SectionHeader icon={<Clock size={16} />} title="Timeouts & Limits" />
          {configLoading ? (
            <Skeleton lines={4} />
          ) : (
            <div className="space-y-3">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {TIMEOUT_FIELDS.map((field) => (
                  <FormInput
                    key={field.key}
                    label={field.label}
                    type="number"
                    value={timeoutValues[field.key] || ""}
                    onChange={(e) =>
                      setTimeoutValues((prev) => ({ ...prev, [field.key]: e.target.value }))
                    }
                    min={0}
                  />
                ))}
              </div>
              <Button onClick={handleSaveTimeouts} loading={timeoutSaving}>
                Save Timeouts & Limits
              </Button>
            </div>
          )}
        </Card>

        {/* ═══ Section 4: Access Tier ═══ */}
        <Card>
          <SectionHeader icon={<Shield size={16} />} title="Access Tier" />
          {configLoading ? (
            <Skeleton lines={2} />
          ) : (
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                {accessTier && (
                  <span className={`rounded-full px-3 py-1 text-xs font-semibold uppercase ${TIER_COLORS[accessTier.tier] || "bg-nova-border text-nova-text-dim"}`}>
                    {accessTier.tier}
                  </span>
                )}
              </div>

              <FormSelect
                label="Change Access Tier"
                value={selectedTier}
                onChange={(e) => setSelectedTier(e.target.value)}
                options={[
                  { value: "sandboxed", label: "Sandboxed" },
                  { value: "standard", label: "Standard" },
                  { value: "full", label: "Full" },
                  { value: "none", label: "None (No restrictions)" },
                ]}
              />

              {selectedTier && TIER_DESCRIPTIONS[selectedTier] && (
                <p className="rounded border border-nova-border bg-nova-bg px-3 py-2 text-xs text-nova-text-dim">
                  {TIER_DESCRIPTIONS[selectedTier]}
                </p>
              )}

              {accessTier && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  <div className="rounded border border-nova-border bg-nova-bg px-3 py-2 text-sm">
                    <span className="text-nova-text-dim">Commands blocked: </span>
                    <span className="font-medium">{accessTier.blocked_commands}</span>
                  </div>
                  <div className="rounded border border-nova-border bg-nova-bg px-3 py-2 text-sm">
                    <span className="text-nova-text-dim">Imports blocked: </span>
                    <span className="font-medium">{accessTier.blocked_imports}</span>
                  </div>
                  <div className="rounded border border-nova-border bg-nova-bg px-3 py-2 text-sm">
                    <span className="text-nova-text-dim">Tool timeout: </span>
                    <span className="font-medium">{accessTier.tool_timeout}s</span>
                  </div>
                  <div className="rounded border border-nova-border bg-nova-bg px-3 py-2 text-sm">
                    <span className="text-nova-text-dim">Gen timeout: </span>
                    <span className="font-medium">{accessTier.generation_timeout}s</span>
                  </div>
                </div>
              )}

              <Button onClick={handleSaveTier} loading={tierSaving}>
                Update Tier
              </Button>
            </div>
          )}
        </Card>

        {/* ═══ Section 5: Theme Toggle (existing) ═══ */}
        <Card>
          <h2 className="mb-3 text-sm font-medium text-nova-text-dim">Theme</h2>
          <div className="flex gap-2">
            {([
              { id: "dark" as const, label: "Dark", icon: <Moon size={16} /> },
              { id: "light" as const, label: "Light", icon: <Sun size={16} /> },
              { id: "system" as const, label: "System", icon: <Monitor size={16} /> },
            ]).map((opt) => (
              <button
                key={opt.id}
                onClick={() => setTheme(opt.id)}
                className={`flex items-center gap-2 rounded-lg border px-4 py-2 text-sm transition-colors ${
                  theme === opt.id
                    ? "border-nova-accent bg-nova-accent/10 text-nova-accent"
                    : "border-nova-border text-nova-text-dim hover:text-nova-text"
                }`}
              >
                {opt.icon}
                {opt.label}
              </button>
            ))}
          </div>
        </Card>

        {/* ═══ API Key (existing) ═══ */}
        <Card>
          <h2 className="mb-3 text-sm font-medium text-nova-text-dim">API Key</h2>
          <div className="flex gap-2">
            <FormInput
              type="password"
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              placeholder="Enter API key (empty = dev mode)"
              icon={<Key size={14} />}
              className="flex-1"
            />
            <Button onClick={saveApiKey}>Save</Button>
          </div>
        </Card>

        {/* ═══ Health (existing) ═══ */}
        <Card>
          <h2 className="mb-3 text-sm font-medium text-nova-text-dim">Health</h2>
          {health ? (
            <div className="grid grid-cols-2 gap-3">
              <StatCard
                label="Status"
                value={health.status}
                className={health.status === "ok" ? "" : "border-nova-warning"}
              />
              <StatCard label="Version" value={health.version} />
              <StatCard label="Model" value={health.model || "\u2014"} />
              <StatCard
                label="LLM"
                value={health.llm_connected ? "connected" : "disconnected"}
              />
            </div>
          ) : (
            <Skeleton lines={2} />
          )}
        </Card>

        {/* ═══ Status / entity counts (existing) ═══ */}
        {status && (
          <Card>
            <h2 className="mb-3 text-sm font-medium text-nova-text-dim">System Status</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {Object.entries(status).map(([k, v]) => (
                <StatCard
                  key={k}
                  label={k.replace(/_/g, " ")}
                  value={v}
                />
              ))}
            </div>
          </Card>
        )}

        {/* ═══ Custom Tools Manager (TASK 6) ═══ */}
        <Card>
          <SectionHeader icon={<Wrench size={16} />} title="Custom Tools" />
          {customToolsLoading ? (
            <Skeleton lines={3} />
          ) : customTools.length === 0 ? (
            <EmptyState
              icon={<Wrench size={40} strokeWidth={1.5} />}
              title="No custom tools"
              description="Custom tools are created automatically when Nova learns new tool patterns."
            />
          ) : (
            <div className="overflow-x-auto rounded-lg border border-nova-border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-nova-border bg-nova-surface text-left text-xs text-nova-text-dim">
                    <th className="px-3 py-2">Name</th>
                    <th className="px-3 py-2">Description</th>
                    <th className="px-3 py-2 text-center">Used</th>
                    <th className="px-3 py-2 text-center">Success</th>
                    <th className="px-3 py-2 text-center">Status</th>
                    <th className="px-3 py-2 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {customTools.map((tool) => (
                    <tr
                      key={tool.name}
                      className="border-b border-nova-border last:border-0 hover:bg-nova-surface/50"
                    >
                      <td className="px-3 py-2 font-medium">{tool.name}</td>
                      <td className="max-w-[200px] truncate px-3 py-2 text-nova-text-dim">
                        {tool.description || "\u2014"}
                      </td>
                      <td className="px-3 py-2 text-center">{tool.times_used}</td>
                      <td className="px-3 py-2 text-center">
                        {Math.round(tool.success_rate * 100)}%
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span className={`rounded px-2 py-0.5 text-xs font-medium ${
                          tool.enabled
                            ? "bg-nova-success/20 text-nova-success"
                            : "bg-nova-error/20 text-nova-error"
                        }`}>
                          {tool.enabled ? "ON" : "OFF"}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setDeletingTool(tool.name)}
                          icon={<Trash2 size={14} />}
                        >
                          Delete
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        {/* ═══ Integrations (existing) ═══ */}
        <Card>
          <h2 className="mb-3 text-sm font-medium text-nova-text-dim">Integrations</h2>
          {integrations === null ? (
            <Skeleton lines={3} />
          ) : integrations.length === 0 ? (
            <p className="text-sm text-nova-text-dim">No integration templates found.</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {integrations.map((i) => (
                <div
                  key={i.name}
                  className="rounded border border-nova-border bg-nova-bg px-4 py-3 animate-fade-in"
                >
                  <div className="flex items-center gap-2">
                    <span className={`h-2 w-2 rounded-full ${i.is_configured ? "bg-nova-success" : "bg-nova-error"}`} />
                    <span className="text-sm font-medium capitalize">{i.name}</span>
                    <span className="ml-auto text-[10px] text-nova-text-dim">
                      {i.endpoint_count} endpoint{i.endpoint_count !== 1 ? "s" : ""}
                    </span>
                  </div>
                  {i.description && (
                    <p className="mt-1 text-xs text-nova-text-dim">{i.description}</p>
                  )}
                  {!i.is_configured && i.auth_env_var && (
                    <p className="mt-1 text-[10px] text-nova-warning">
                      Set <code className="rounded bg-nova-border px-1">{i.auth_env_var}</code> to enable
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* ═══ User Facts (existing) ═══ */}
        <Card>
          <h2 className="mb-3 text-sm font-medium text-nova-text-dim">User Facts</h2>
          <div className="flex gap-2">
            <FormInput
              value={newKey}
              onChange={(e) => setNewKey(e.target.value)}
              placeholder="Key"
              className="w-1/3"
            />
            <FormInput
              value={newValue}
              onChange={(e) => setNewValue(e.target.value)}
              placeholder="Value"
              className="flex-1"
            />
            <Button onClick={handleCreateFact}>Add</Button>
          </div>
          {facts.length === 0 && (
            <p className="mt-2 text-sm text-nova-text-dim">No facts stored yet.</p>
          )}
          <div className="mt-2 space-y-1">
            {facts.map((f) => (
              <div
                key={f.key}
                className="flex items-center justify-between rounded border border-nova-border bg-nova-bg px-3 py-2 text-sm"
              >
                <div>
                  <span className="font-medium">{f.key}</span>
                  <span className="text-nova-text-dim"> = </span>
                  {f.value}
                </div>
                <Button variant="ghost" size="sm" onClick={() => handleDeleteFact(f.key)}>
                  Delete
                </Button>
              </div>
            ))}
          </div>
        </Card>

        {/* ═══ Export / Import (existing) ═══ */}
        <Card>
          <h2 className="mb-3 text-sm font-medium text-nova-text-dim">Data</h2>
          <div className="flex gap-3">
            <Button variant="secondary" onClick={handleExport} icon={<Download size={16} />}>
              Export JSON
            </Button>
            <Button variant="secondary" onClick={handleImport} icon={<Upload size={16} />}>
              Import JSON
            </Button>
          </div>
        </Card>

        {/* ═══ Confirm dialog for custom tool delete ═══ */}
        {deletingTool !== null && (
          <ConfirmDialog
            message={`Delete custom tool "${deletingTool}"? This cannot be undone.`}
            onConfirm={handleDeleteTool}
            onCancel={() => setDeletingTool(null)}
          />
        )}
      </div>
    </div>
  );
}
