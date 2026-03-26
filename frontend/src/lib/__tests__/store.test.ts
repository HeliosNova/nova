import { describe, it, expect, beforeEach } from "vitest";
import { useChatStore } from "../store";

describe("Chat store", () => {
  beforeEach(() => {
    // Reset store to initial state before each test
    useChatStore.setState({
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
    });
  });

  describe("newChat", () => {
    it("resets activeConversationId to null", () => {
      useChatStore.setState({ activeConversationId: "conv-123" });
      useChatStore.getState().newChat();
      expect(useChatStore.getState().activeConversationId).toBeNull();
    });

    it("clears all messages", () => {
      useChatStore.setState({
        messages: [
          { role: "user", content: "Hello" },
          { role: "assistant", content: "Hi there!" },
        ],
      });
      useChatStore.getState().newChat();
      expect(useChatStore.getState().messages).toEqual([]);
    });

    it("resets streaming state", () => {
      useChatStore.setState({
        streaming: true,
        streamedTokens: "partial response...",
        thinkingStage: "analyzing",
        thinkingContent: "thinking...",
      });
      useChatStore.getState().newChat();
      const state = useChatStore.getState();
      expect(state.streaming).toBe(false);
      expect(state.streamedTokens).toBe("");
      expect(state.thinkingStage).toBeNull();
      expect(state.thinkingContent).toBeNull();
    });

    it("clears tool calls and pending data", () => {
      useChatStore.setState({
        toolCalls: [{ id: "tc-1", tool: "web_search", status: "executing" }],
        pendingLessonsUsed: [{ topic: "test", confidence: 0.9, lesson_id: 1 }],
        pendingLessonLearned: { topic: "test", lesson_id: 1 },
        pendingSources: [{ title: "source" }],
      });
      useChatStore.getState().newChat();
      const state = useChatStore.getState();
      expect(state.toolCalls).toEqual([]);
      expect(state.pendingLessonsUsed).toEqual([]);
      expect(state.pendingLessonLearned).toBeNull();
      expect(state.pendingSources).toEqual([]);
    });

    it("clears error state", () => {
      useChatStore.setState({ error: "Something went wrong" });
      useChatStore.getState().newChat();
      expect(useChatStore.getState().error).toBeNull();
    });
  });

  describe("addUserMessage", () => {
    it("adds a user message to the messages array", () => {
      useChatStore.getState().addUserMessage("Hello!");
      const messages = useChatStore.getState().messages;
      expect(messages).toHaveLength(1);
      expect(messages[0].role).toBe("user");
      expect(messages[0].content).toBe("Hello!");
    });

    it("appends to existing messages", () => {
      useChatStore.setState({
        messages: [{ role: "assistant", content: "Welcome!" }],
      });
      useChatStore.getState().addUserMessage("Thanks");
      expect(useChatStore.getState().messages).toHaveLength(2);
    });

    it("includes image_base64 when provided", () => {
      useChatStore.getState().addUserMessage("Look at this", "data:image/png;base64,abc123");
      const msg = useChatStore.getState().messages[0];
      expect(msg.image_base64).toBe("data:image/png;base64,abc123");
    });
  });

  describe("toggleSidebar", () => {
    it("toggles the sidebar open state", () => {
      const initial = useChatStore.getState().sidebarOpen;
      useChatStore.getState().toggleSidebar();
      expect(useChatStore.getState().sidebarOpen).toBe(!initial);
      useChatStore.getState().toggleSidebar();
      expect(useChatStore.getState().sidebarOpen).toBe(initial);
    });
  });
});
