import { create } from "zustand";
import type {
  ConversationSummary,
  Message,
  ToolCall,
  HealthResponse,
  LearningMetrics,
  LessonInfo,
  SkillInfo,
} from "./types";

// ── Chat Store ──

interface ChatState {
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  messages: Message[];
  streaming: boolean;
  thinkingStage: string | null;
  thinkingContent: string | null;
  streamedTokens: string;
  toolCalls: ToolCall[];
  pendingLessonsUsed: { topic: string; confidence: number; lesson_id: number }[];
  pendingLessonLearned: { topic: string; lesson_id: number } | null;
  pendingSources: Record<string, unknown>[];
  error: string | null;
  sidebarOpen: boolean;

  setConversations: (convs: ConversationSummary[]) => void;
  setActiveConversation: (id: string | null, messages: Message[]) => void;
  addUserMessage: (content: string, image_base64?: string) => void;
  addLessonUsed: (lesson: { topic: string; confidence: number; lesson_id: number }) => void;
  setLessonLearned: (lesson: { topic: string; lesson_id: number }) => void;
  setSources: (sources: Record<string, unknown>[]) => void;
  setStreaming: (s: boolean) => void;
  setThinkingStage: (stage: string | null) => void;
  setThinkingContent: (content: string | null) => void;
  appendToken: (text: string) => void;
  addToolCall: (tc: ToolCall) => void;
  updateToolCall: (id: string, updates: Partial<ToolCall>) => void;
  finalizeAssistantMessage: (conversationId: string) => void;
  setError: (err: string | null) => void;
  resetStream: () => void;
  newChat: () => void;
  toggleSidebar: () => void;
  removeLastAssistantMessage: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  conversations: [],
  activeConversationId: null,
  messages: [],
  streaming: false,
  thinkingStage: null,
  thinkingContent: null,
  streamedTokens: "",
  toolCalls: [],
  pendingLessonsUsed: [],
  pendingLessonLearned: null,
  pendingSources: [],
  error: null,
  sidebarOpen: true,

  setConversations: (conversations) => set({ conversations }),

  setActiveConversation: (id, messages) =>
    set({
      activeConversationId: id,
      messages,
      streamedTokens: "",
      toolCalls: [],
      pendingLessonsUsed: [],
      pendingLessonLearned: null,
      pendingSources: [],
      thinkingStage: null,
      thinkingContent: null,
      error: null,
    }),

  addUserMessage: (content, image_base64) =>
    set((s) => ({
      messages: [...s.messages, { role: "user" as const, content, image_base64 }],
    })),

  addLessonUsed: (lesson) =>
    set((s) => ({ pendingLessonsUsed: [...s.pendingLessonsUsed, lesson] })),

  setLessonLearned: (lesson) =>
    set({ pendingLessonLearned: lesson }),

  setSources: (pendingSources) => set({ pendingSources }),

  setStreaming: (streaming) => set({ streaming }),
  setThinkingStage: (thinkingStage) => set({ thinkingStage }),
  setThinkingContent: (thinkingContent) =>
    set((s) => ({
      thinkingContent: thinkingContent === null
        ? null
        : (s.thinkingContent || "") + thinkingContent,
    })),

  appendToken: (text) =>
    set((s) => ({ streamedTokens: s.streamedTokens + text })),

  addToolCall: (tc) =>
    set((s) => ({ toolCalls: [...s.toolCalls, tc] })),

  updateToolCall: (id, updates) =>
    set((s) => ({
      toolCalls: s.toolCalls.map((tc) =>
        tc.id === id ? { ...tc, ...updates } : tc
      ),
    })),

  finalizeAssistantMessage: (conversationId) =>
    set((s) => ({
      activeConversationId: conversationId,
      messages: [
        ...s.messages,
        {
          role: "assistant" as const,
          content: s.streamedTokens,
          lessons_used: s.pendingLessonsUsed.length > 0 ? s.pendingLessonsUsed : undefined,
          lesson_learned: s.pendingLessonLearned || undefined,
          sources: s.pendingSources.length > 0 ? s.pendingSources : undefined,
        },
      ],
      streamedTokens: "",
      toolCalls: [],
      pendingLessonsUsed: [],
      pendingLessonLearned: null,
      pendingSources: [],
      thinkingStage: null,
      thinkingContent: null,
      streaming: false,
    })),

  setError: (error) => set({ error, streaming: false }),

  resetStream: () =>
    set({ streamedTokens: "", toolCalls: [], pendingLessonsUsed: [], pendingLessonLearned: null, pendingSources: [], thinkingStage: null, thinkingContent: null, error: null }),

  newChat: () =>
    set({
      activeConversationId: null,
      messages: [],
      streamedTokens: "",
      toolCalls: [],
      pendingLessonsUsed: [],
      pendingLessonLearned: null,
      pendingSources: [],
      thinkingStage: null,
      thinkingContent: null,
      error: null,
      streaming: false,
    }),

  toggleSidebar: () =>
    set((s) => ({ sidebarOpen: !s.sidebarOpen })),

  removeLastAssistantMessage: () =>
    set((s) => {
      const msgs = [...s.messages];
      for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === "assistant") {
          msgs.splice(i, 1);
          break;
        }
      }
      return { messages: msgs };
    }),
}));

// ── Learning Store ──

interface LearningState {
  metrics: LearningMetrics | null;
  lessons: LessonInfo[];
  skills: SkillInfo[];
  loading: boolean;

  setMetrics: (m: LearningMetrics) => void;
  setLessons: (l: LessonInfo[]) => void;
  setSkills: (s: SkillInfo[]) => void;
  removeLesson: (id: number) => void;
  removeSkill: (id: number) => void;
  updateSkillEnabled: (id: number, enabled: boolean) => void;
  setLoading: (l: boolean) => void;
}

export const useLearningStore = create<LearningState>((set) => ({
  metrics: null,
  lessons: [],
  skills: [],
  loading: false,

  setMetrics: (metrics) => set({ metrics }),
  setLessons: (lessons) => set({ lessons }),
  setSkills: (skills) => set({ skills }),
  removeLesson: (id) =>
    set((s) => ({ lessons: s.lessons.filter((l) => l.id !== id) })),
  removeSkill: (id) =>
    set((s) => ({ skills: s.skills.filter((sk) => sk.id !== id) })),
  updateSkillEnabled: (id, enabled) =>
    set((s) => ({
      skills: s.skills.map((sk) => (sk.id === id ? { ...sk, enabled } : sk)),
    })),
  setLoading: (loading) => set({ loading }),
}));

// ── Settings Store ──

type ThemeSetting = "dark" | "light" | "system";

interface SettingsState {
  health: HealthResponse | null;
  apiKey: string;
  theme: ThemeSetting;

  setHealth: (h: HealthResponse) => void;
  setApiKey: (key: string) => void;
  setTheme: (t: ThemeSetting) => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  health: null,
  apiKey: localStorage.getItem("nova_api_key") || "",
  theme: (localStorage.getItem("nova_theme") as ThemeSetting) || "dark",

  setHealth: (health) => set({ health }),
  setApiKey: (apiKey) => {
    localStorage.setItem("nova_api_key", apiKey);
    set({ apiKey });
  },
  setTheme: (theme) => {
    localStorage.setItem("nova_theme", theme);
    set({ theme });
  },
}));
