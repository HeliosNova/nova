import type {
  ConversationSummary,
  ConversationDetail,
  HealthResponse,
  StatusResponse,
  LearningMetrics,
  LessonInfo,
  SkillInfo,
  UserFact,
  UserFactCreate,
  DocumentInfo,
  DocumentSearchResult,
  MonitorInfo,
  MonitorCreate,
  MonitorResult,
  MonitorDetail,
  ActionInfo,
  ReflexionInfo,
  TrainingDataStats,
  TrainingDataEntry,
  TrainingDataStatsDetailed,
  FinetuneStatus,
  FinetuneHistoryRun,
  MessageSearchResult,
  IntegrationInfo,
  AccessTierInfo,
  ConfigSummary,
  FullConfig,
  ConfigUpdateResponse,
  KGFact,
  CustomToolInfo,
  CuriosityItem,
  FinetuneTriggerResponse,
} from "./types";

export function getBaseUrl(): string {
  // In dev, Vite proxy handles /api -> backend. In prod, same origin.
  return "";
}

function getHeaders(): HeadersInit {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  const apiKey = localStorage.getItem("nova_api_key");
  if (apiKey) {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }
  return headers;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${getBaseUrl()}${path}`, {
    ...init,
    headers: { ...getHeaders(), ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

/** Safely extract an array from a response that may be a bare array or { key: [...] } */
function ensureArray<T>(data: unknown, key?: string): T[] {
  if (Array.isArray(data)) return data;
  if (data && typeof data === "object") {
    if (key && Array.isArray((data as Record<string, unknown>)[key])) {
      return (data as Record<string, unknown>)[key] as T[];
    }
    // Fallback: find first array property
    for (const v of Object.values(data as Record<string, unknown>)) {
      if (Array.isArray(v)) return v as T[];
    }
  }
  return [];
}

// ── System ──

export async function getHealth(): Promise<HealthResponse> {
  return request("/api/health");
}

export async function getStatus(): Promise<StatusResponse> {
  return request("/api/status");
}

export async function exportData(): Promise<Blob> {
  const res = await fetch(`${getBaseUrl()}/api/export`, { headers: getHeaders() });
  if (!res.ok) throw new Error(`Export failed: ${res.status}`);
  return res.blob();
}

export async function importData(file: File): Promise<{ status: string; stats: Record<string, number> }> {
  const form = new FormData();
  form.append("file", file);
  const headers: Record<string, string> = {};
  const apiKey = localStorage.getItem("nova_api_key");
  if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;
  const res = await fetch(`${getBaseUrl()}/api/import`, {
    method: "POST",
    headers,
    body: form,
  });
  if (!res.ok) throw new Error(`Import failed: ${res.status}`);
  return res.json();
}

// ── Voice ──

export async function transcribeVoice(audioBlob: Blob, filename = "recording.webm"): Promise<{ text: string; language?: string; duration?: number }> {
  const form = new FormData();
  form.append("file", audioBlob, filename);
  const headers: Record<string, string> = {};
  const apiKey = localStorage.getItem("nova_api_key");
  if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;
  const res = await fetch(`${getBaseUrl()}/api/voice/transcribe`, {
    method: "POST",
    headers,
    body: form,
  });
  if (!res.ok) throw new Error(`Transcription failed: ${res.status}`);
  return res.json();
}

// ── Conversations ──

export async function getConversations(limit = 50): Promise<ConversationSummary[]> {
  const res = await request(`/api/chat/conversations?limit=${limit}`);
  return ensureArray<ConversationSummary>(res, "conversations");
}

export async function getConversation(id: string): Promise<ConversationDetail> {
  return request(`/api/chat/conversations/${id}`);
}

export async function deleteConversation(id: string): Promise<void> {
  await request(`/api/chat/conversations/${id}`, { method: "DELETE" });
}

export async function searchConversations(q: string): Promise<ConversationSummary[]> {
  const res = await request(`/api/chat/conversations/search?q=${encodeURIComponent(q)}`);
  return ensureArray<ConversationSummary>(res, "conversations");
}

export async function renameConversation(id: string, title: string): Promise<void> {
  await request(`/api/chat/conversations/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

export async function searchMessages(query: string): Promise<MessageSearchResult[]> {
  const res = await request(`/api/chat/messages/search?q=${encodeURIComponent(query)}`);
  return ensureArray<MessageSearchResult>(res, "messages");
}

// ── User Facts ──

export async function getFacts(): Promise<UserFact[]> {
  const res = await request("/api/chat/facts");
  return ensureArray<UserFact>(res, "facts");
}

export async function createFact(fact: UserFactCreate): Promise<{ status: string; key: string }> {
  return request("/api/chat/facts", {
    method: "POST",
    body: JSON.stringify(fact),
  });
}

export async function deleteFact(key: string): Promise<void> {
  await request(`/api/chat/facts/${encodeURIComponent(key)}`, { method: "DELETE" });
}

// ── Learning ──

export async function getLearningMetrics(): Promise<LearningMetrics> {
  return request("/api/learning/metrics");
}

export async function getLessons(limit = 100): Promise<LessonInfo[]> {
  const res = await request(`/api/learning/lessons?limit=${limit}`);
  return ensureArray<LessonInfo>(res, "lessons");
}

export async function deleteLesson(id: number): Promise<void> {
  await request(`/api/learning/lessons/${id}`, { method: "DELETE" });
}

export async function bulkDeleteLessons(ids: number[]): Promise<void> {
  await request("/api/learning/lessons/bulk-delete", {
    method: "POST",
    body: JSON.stringify({ ids }),
  });
}

export async function getSkills(limit = 50): Promise<SkillInfo[]> {
  const res = await request(`/api/learning/skills?limit=${limit}`);
  return ensureArray<SkillInfo>(res, "skills");
}

export async function toggleSkill(id: number, enabled: boolean): Promise<void> {
  await request(`/api/learning/skills/${id}/toggle?enabled=${enabled}`, { method: "POST" });
}

export async function deleteSkill(id: number): Promise<void> {
  await request(`/api/learning/skills/${id}`, { method: "DELETE" });
}

// ── Documents ──

export async function getDocuments(): Promise<DocumentInfo[]> {
  const res = await request("/api/documents");
  return ensureArray<DocumentInfo>(res, "documents");
}

export async function deleteDocument(id: number): Promise<void> {
  await request(`/api/documents/${id}`, { method: "DELETE" });
}

export async function ingestDocument(data: { text?: string; url?: string; title?: string }): Promise<{ status: string; document_id: number }> {
  return request("/api/documents/ingest", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function searchDocuments(query: string): Promise<DocumentSearchResult[]> {
  const res = await request("/api/documents/search", {
    method: "POST",
    body: JSON.stringify({ query }),
  });
  return ensureArray<DocumentSearchResult>(res, "results");
}

// ── Monitors ──

export async function getMonitors(): Promise<MonitorInfo[]> {
  const res = await request("/api/monitors");
  return ensureArray<MonitorInfo>(res, "monitors");
}

export async function createMonitor(data: MonitorCreate): Promise<MonitorInfo> {
  return request("/api/monitors", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function updateMonitor(id: number, data: Partial<MonitorCreate>): Promise<void> {
  await request(`/api/monitors/${id}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export async function deleteMonitor(id: number): Promise<void> {
  await request(`/api/monitors/${id}`, { method: "DELETE" });
}

export async function triggerMonitor(id: number): Promise<{ status: string; result: unknown }> {
  return request(`/api/monitors/${id}/trigger`, { method: "POST" });
}

export async function getRecentMonitorResults(hours = 24): Promise<MonitorResult[]> {
  const res = await request(`/api/monitors/results/recent?hours=${hours}`);
  return ensureArray<MonitorResult>(res, "results");
}

export async function getMonitorDetail(id: number): Promise<MonitorDetail> {
  return request(`/api/monitors/${id}`);
}

export async function rateMonitorResult(resultId: number, rating: -1 | 0 | 1): Promise<void> {
  await request(`/api/monitors/results/${resultId}/rate`, {
    method: "POST",
    body: JSON.stringify({ rating }),
  });
}

// ── Heartbeat Instructions ──

export async function getHeartbeatInstructions(): Promise<unknown[]> {
  const res = await request("/api/heartbeat/instructions");
  return (res as { instructions?: unknown[] })?.instructions || [];
}

export async function createHeartbeatInstruction(data: { instruction: string; schedule_seconds: number; notify_channels?: string }): Promise<unknown> {
  return request("/api/heartbeat/instructions", { method: "POST", body: JSON.stringify(data) });
}

export async function deleteHeartbeatInstruction(id: number): Promise<void> {
  await request(`/api/heartbeat/instructions/${id}`, { method: "DELETE" });
}

// ── Actions ──

export async function getActions(type?: string, hours = 24, limit = 50): Promise<ActionInfo[]> {
  let path = `/api/actions?hours=${hours}&limit=${limit}`;
  if (type) path += `&action_type=${encodeURIComponent(type)}`;
  const res = await request(path);
  return ensureArray<ActionInfo>(res, "actions");
}

// ── Reflexions ──

export async function getReflexions(): Promise<ReflexionInfo[]> {
  const res = await request("/api/learning/reflexions");
  return ensureArray<ReflexionInfo>(res, "reflexions");
}

// ── Training Data ──

export async function getTrainingDataStats(): Promise<TrainingDataStats> {
  return request("/api/learning/training-data/stats");
}

export async function getTrainingData(limit = 50, offset = 0): Promise<TrainingDataEntry[]> {
  const res = await request(`/api/learning/training-data?limit=${limit}&offset=${offset}`);
  return ensureArray<TrainingDataEntry>(res, "entries");
}

export async function exportTrainingData(): Promise<Blob> {
  const res = await fetch(`${getBaseUrl()}/api/learning/training-data/export`, {
    method: "POST",
    headers: getHeaders(),
  });
  if (!res.ok) throw new Error(`Export failed: ${res.status}`);
  return res.blob();
}

export async function getFinetuneStatus(): Promise<FinetuneStatus> {
  return request("/api/learning/finetune/status");
}

export async function getFinetuneHistory(): Promise<FinetuneHistoryRun[]> {
  const res = await request("/api/learning/finetune/history");
  return ensureArray<FinetuneHistoryRun>(res, "runs");
}

export async function getTrainingDataStatsDetailed(): Promise<TrainingDataStatsDetailed> {
  return request("/api/learning/training-data/stats");
}

// ── Integrations / Access Tier / Config ──

export async function getIntegrations(): Promise<IntegrationInfo[]> {
  const res = await request("/api/integrations");
  return ensureArray<IntegrationInfo>(res, "integrations");
}

export async function getAccessTier(): Promise<AccessTierInfo> {
  return request("/api/access-tier");
}

export async function getConfigSummary(): Promise<ConfigSummary> {
  return request("/api/config-summary");
}

// ── Full Config / Update ──

export async function getFullConfig(): Promise<FullConfig> {
  return request("/api/config/full");
}

export async function updateConfig(updates: Record<string, unknown>): Promise<ConfigUpdateResponse> {
  return request("/api/config", {
    method: "PATCH",
    body: JSON.stringify(updates),
  });
}

// ── Knowledge Graph ──

export async function getKGFacts(limit = 50, offset = 0, search = ""): Promise<KGFact[]> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (search) params.set("search", search);
  const res = await request(`/api/kg/facts?${params}`);
  return ensureArray<KGFact>(res, "facts");
}

// ── Custom Tools ──

export async function getCustomTools(): Promise<CustomToolInfo[]> {
  const res = await request("/api/custom-tools");
  return ensureArray<CustomToolInfo>(res, "tools");
}

export async function deleteCustomTool(name: string): Promise<void> {
  await request(`/api/custom-tools/${encodeURIComponent(name)}`, { method: "DELETE" });
}

// ── Curiosity ──

export async function getCuriosityQueue(): Promise<CuriosityItem[]> {
  const res = await request("/api/curiosity/queue");
  return ensureArray<CuriosityItem>(res, "items");
}

// ── Fine-tune Trigger ──

export async function triggerFinetune(): Promise<FinetuneTriggerResponse> {
  return request("/api/learning/finetune/trigger", { method: "POST" });
}

// ── Heartbeat Instructions (single) ──

export async function getHeartbeatInstruction(id: number): Promise<unknown> {
  return request(`/api/monitors/${id}/instruction`);
}

export async function updateHeartbeatInstruction(id: number, data: object): Promise<unknown> {
  return request(`/api/monitors/${id}/instruction`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

// ── Actions (single) ──

export async function getAction(id: number): Promise<ActionInfo> {
  return request(`/api/actions/${id}`);
}

// ── Documents (single) ──

export async function getDocument(id: number): Promise<DocumentInfo> {
  return request(`/api/documents/${id}`);
}

// ── Voice Chat ──

export async function voiceChat(blob: Blob, filename = "recording.webm"): Promise<Response> {
  const form = new FormData();
  form.append("file", blob, filename);
  const headers: Record<string, string> = {};
  const apiKey = localStorage.getItem("nova_api_key");
  if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;
  const res = await fetch(`${getBaseUrl()}/api/voice/chat`, {
    method: "POST",
    headers,
    body: form,
  });
  if (!res.ok) throw new Error(`Voice chat failed: ${res.status}`);
  return res;
}
