// ── Chat ──

export interface ChatRequest {
  query: string;
  conversation_id?: string | null;
  image_base64?: string | null;
}

export interface ChatResponse {
  answer: string;
  conversation_id: string;
  sources: Record<string, unknown>[];
  tool_results: ToolResult[];
  lessons_used: number;
  skill_used: string | null;
}

export interface ToolResult {
  tool: string;
  result: string;
}

// ── SSE Events ──

export type EventType = "thinking" | "token" | "tool_use" | "sources" | "lesson_used" | "lesson_learned" | "warning" | "done" | "error";

export interface SSEThinkingData {
  stage?: string;
  content?: string;
}

export interface SSETokenData {
  text: string;
}

export interface SSEToolUseData {
  tool: string;
  status: "executing" | "complete";
  result?: string;
  tool_call_id?: string;
  args?: Record<string, unknown>;
}

export interface SSESourcesData {
  sources: Record<string, unknown>[];
}

export interface SSEDoneData {
  conversation_id: string;
  lessons_used: number;
  skill_used: string | null;
}

export interface SSEErrorData {
  message: string;
}

export interface SSELessonUsedData {
  topic: string;
  confidence: number;
  lesson_id: number;
}

export interface SSELessonLearnedData {
  topic: string;
  lesson_id: number;
}

export interface StreamEvent {
  type: EventType;
  data: SSEThinkingData | SSETokenData | SSEToolUseData | SSESourcesData | SSEDoneData | SSEErrorData;
}

// ── Conversations ──

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id?: number;
  role: "user" | "assistant";
  content: string;
  created_at?: string;
  image_base64?: string;
  lessons_used?: { topic: string; confidence: number; lesson_id: number }[];
  lesson_learned?: { topic: string; lesson_id: number };
  sources?: Record<string, unknown>[];
}

// ── Message Search Result ──

export interface MessageSearchResult {
  conversation_id: string;
  role: string;
  content: string;
  created_at: string;
}

export interface ConversationDetail {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: Message[];
}

// ── User Facts ──

export interface UserFact {
  id: number | null;
  key: string;
  value: string;
  source: string;
  confidence: number;
  category?: string;
}

export interface UserFactCreate {
  key: string;
  value: string;
  source?: string;
  category?: string;
}

// ── Learning ──

export interface LearningMetrics {
  total_lessons: number;
  total_skills: number;
  total_corrections: number;
  training_examples: number;
  last_correction_date: string | null;
}

export interface LessonInfo {
  id: number;
  topic: string;
  wrong_answer: string | null;
  correct_answer: string;
  lesson_text?: string;
  confidence: number;
  times_retrieved: number;
  times_helpful: number;
  created_at: string;
}

export interface SkillInfo {
  id: number;
  name: string;
  trigger_pattern: string;
  steps: Record<string, unknown>[];
  answer_template?: string;
  times_used: number;
  success_rate: number;
  enabled: boolean;
  created_at?: string;
}

// ── System ──

export interface HealthResponse {
  status: string;
  timestamp: string;
}

export interface StatusResponse {
  conversations: number;
  messages: number;
  user_facts: number;
  lessons: number;
  skills: number;
  documents: number;
  training_examples: number;
  kg_facts: number;
  reflexions: number;
  custom_tools: number;
}

// ── Tool Call (UI state) ──

export interface ToolCall {
  id: string;
  tool: string;
  status: "executing" | "complete";
  result?: string;
  tool_call_id?: string;
  args?: Record<string, unknown>;
}

// ── Documents ──

export interface DocumentInfo {
  id: number;
  title: string;
  source: string;
  chunk_count: number;
  created_at: string;
}

export interface DocumentSearchResult {
  content: string;
  title: string;
  source: string;
  score: number;
}

// ── Monitors ──

export interface MonitorInfo {
  id: number;
  name: string;
  check_type: string;
  check_config: Record<string, unknown>;
  schedule_seconds: number;
  enabled: boolean;
  cooldown_minutes: number;
  notify_condition: string;
  last_check_at: string | null;
  last_alert_at: string | null;
  last_result: string | null;
  created_at: string;
}

export interface MonitorCreate {
  name: string;
  check_type: string;
  check_config: Record<string, unknown>;
  schedule_seconds: number;
  cooldown_minutes?: number;
  notify_condition?: string;
}

export interface MonitorResult {
  id: number;
  monitor_id: number;
  status: string;
  value: string | null;
  message: string | null;
  created_at: string;
}

// ── Actions ──

export interface ActionInfo {
  id: number;
  action_type: string;
  params: string | null;
  result: string | null;
  success: boolean;
  created_at: string;
}

// ── Reflexions ──

export interface ReflexionInfo {
  id: number;
  task_summary: string;
  outcome: string;
  reflection: string;
  quality_score: number;
  tools_used: string[];
  created_at: string;
}

// ── Training Data ──

export interface TrainingDataStats {
  total_pairs: number;
  ready_for_export: boolean;
}

export interface TrainingDataEntry {
  id: number;
  query: string;
  chosen: string;
  rejected: string;
  created_at: string;
}

export interface FinetuneStatus {
  training_pairs: number;
  ready: boolean;
  min_required: number;
  total_pairs?: number;
  valid_pairs?: number;
  recommendation?: string;
}

// ── Fine-tune History ──

export interface FinetuneHistoryRun {
  started_at: string | null;
  completed_at: string | null;
  status: string | null;
  training_pairs: number;
  new_pairs: number;
  base_model: string | null;
  ft_model: string | null;
  reason?: string;
  eval?: {
    win_rate: number | null;
    avg_preference: number | null;
    candidate_wins: number | null;
    base_wins: number | null;
    candidate_is_better: boolean | null;
  };
}

// ── Training Data Stats (detailed) ──

export interface TrainingDataStatsDetailed {
  total_pairs: number;
  valid_pairs: number;
  topics: string[];
  date_range: { earliest: string | null; latest: string | null } | null;
  avg_chosen_length: number;
  avg_rejected_length: number;
}

// ── Monitor Detail (with results) ──

export interface MonitorDetail extends MonitorInfo {
  results: {
    id: number;
    status: string;
    value: string | null;
    message: string | null;
    created_at: string;
  }[];
}

// ── Integrations ──

export interface IntegrationInfo {
  name: string;
  auth_type: string;
  auth_env_var: string;
  is_configured: boolean;
  endpoint_count: number;
  description: string;
}

// ── Access Tier ──

export interface AccessTierInfo {
  tier: string;
  description: string;
  blocked_commands: number;
  blocked_imports: number;
  tool_timeout: number;
  generation_timeout: number;
}

// ── Config Summary ──

export interface ConfigSummary {
  LLM_PROVIDER: string;
  LLM_MODEL: string;
  ENABLE_MCP: boolean;
  ENABLE_DELEGATION: boolean;
  MAX_DELEGATION_DEPTH: number;
  ENABLE_AUTO_SKILL_CREATION: boolean;
  ENABLE_MODEL_ROUTING: boolean;
  ENABLE_EXTENDED_THINKING: boolean;
  TOOL_TIMEOUT: number;
  GENERATION_TIMEOUT: number;
  SYSTEM_ACCESS_LEVEL: string;
}

// ── Full Config (all settings) ──

export interface FullConfig {
  LLM_PROVIDER?: string;
  LLM_MODEL?: string;
  OLLAMA_URL?: string;
  SYSTEM_ACCESS_LEVEL?: string;
  ENABLE_VOICE?: boolean;
  ENABLE_MCP?: boolean;
  ENABLE_DELEGATION?: boolean;
  ENABLE_PLANNING?: boolean;
  ENABLE_CRITIQUE?: boolean;
  ENABLE_CURIOSITY?: boolean;
  ENABLE_HEARTBEAT?: boolean;
  ENABLE_PROACTIVE?: boolean;
  ENABLE_CUSTOM_TOOLS?: boolean;
  ENABLE_EXTENDED_THINKING?: boolean;
  ENABLE_MODEL_ROUTING?: boolean;
  MAX_TOOL_ROUNDS?: number;
  TOOL_TIMEOUT?: number;
  GENERATION_TIMEOUT?: number;
  [key: string]: unknown;
}

// ── Config Update Response ──

export interface ConfigUpdateResponse {
  updated: string[];
  warnings: string[];
  restart_required: boolean;
}

// ── Knowledge Graph Fact ──

export interface KGFact {
  id: number;
  subject: string;
  predicate: string;
  object: string;
  confidence: number;
  source: string;
  created_at: string;
  valid_from?: string;
  valid_to?: string | null;
}

// ── Custom Tool ──

export interface CustomToolInfo {
  name: string;
  description: string;
  times_used: number;
  success_rate: number;
  enabled: boolean;
  created_at?: string;
}

// ── Curiosity Queue Item ──

export interface CuriosityItem {
  id: number;
  question: string;
  source: string;
  priority: number;
  status: string;
  created_at: string;
}

// ── Fine-tune Trigger Response ──

export interface FinetuneTriggerResponse {
  status: string;
  message?: string;
}
