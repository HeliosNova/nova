import { useState, useRef, useEffect, useCallback } from "react";
import { useInView } from "react-intersection-observer";
import { MessageSquare, RotateCcw, ChevronDown } from "lucide-react";
import { useChatStore } from "../lib/store";
import { getConversation, getConversations } from "../lib/api";
import { streamChat } from "../lib/sse";
import { uid } from "../lib/utils";
import type { StreamEvent, SSETokenData, SSEToolUseData, SSESourcesData, SSEDoneData, SSEErrorData, SSEThinkingData, SSELessonUsedData, SSELessonLearnedData } from "../lib/types";
import ChatSidebar from "../components/ChatSidebar";
import ChatMessage from "../components/ChatMessage";
import ChatInput from "../components/ChatInput";
import ThinkingIndicator from "../components/ThinkingIndicator";
import ToolCallCard from "../components/ToolCallCard";
import { ErrorBanner } from "../components/ui";

// Helper to get store actions without causing re-renders
const actions = () => useChatStore.getState();

export default function ChatPage() {
  // Select individual fields to avoid re-rendering on unrelated state changes
  const messages = useChatStore((s) => s.messages);
  const streaming = useChatStore((s) => s.streaming);
  const streamedTokens = useChatStore((s) => s.streamedTokens);
  const toolCalls = useChatStore((s) => s.toolCalls);
  const thinkingStage = useChatStore((s) => s.thinkingStage);
  const thinkingContent = useChatStore((s) => s.thinkingContent);
  const activeConversationId = useChatStore((s) => s.activeConversationId);
  const error = useChatStore((s) => s.error);

  const abortRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const userScrolledUpRef = useRef(false);

  // Intersection observer sentinel — replaces manual scroll tracking
  const { ref: bottomSentinelRef, inView: isAtBottom } = useInView({
    threshold: 0,
  });

  // Update scroll state when sentinel visibility changes
  useEffect(() => {
    userScrolledUpRef.current = !isAtBottom;
    setShowScrollBtn(!isAtBottom);
  }, [isAtBottom]);

  const scrollToBottom = useCallback((force = false) => {
    if (force || !userScrolledUpRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, []);

  // Auto-scroll only when near bottom
  useEffect(() => {
    if (!userScrolledUpRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, streamedTokens, toolCalls]);

  const loadConversation = useCallback(async (id: string) => {
    try {
      const conv = await getConversation(id);
      actions().setActiveConversation(id, conv.messages);
    } catch {
      actions().setError("Failed to load conversation");
    }
  }, []);

  const handleSend = useCallback(async (text: string, imageBase64?: string) => {
    const s = actions();
    s.addUserMessage(text, imageBase64);
    s.resetStream();
    s.setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    await streamChat(text, useChatStore.getState().activeConversationId, {
      signal: controller.signal,
      imageBase64,
      onEvent: (event: StreamEvent) => {
        const a = actions();
        switch (event.type) {
          case "thinking": {
            const d = event.data as SSEThinkingData;
            a.setThinkingStage(d.stage || "thinking");
            if (d.stage === "reasoning" && d.content) {
              a.setThinkingContent(d.content);
            }
            break;
          }
          case "token": {
            const d = event.data as SSETokenData;
            a.setThinkingStage(null);
            a.appendToken(d.text);
            break;
          }
          case "tool_use": {
            const d = event.data as SSEToolUseData;
            if (d.status === "executing") {
              a.addToolCall({
                id: uid(),
                tool: d.tool,
                status: "executing",
              });
            } else {
              const tc = useChatStore
                .getState()
                .toolCalls.find(
                  (t) => t.tool === d.tool && t.status === "executing"
                );
              if (tc) {
                a.updateToolCall(tc.id, {
                  status: "complete",
                  result: d.result,
                });
              }
            }
            break;
          }
          case "sources": {
            const d = event.data as SSESourcesData;
            if (d.sources) a.setSources(d.sources);
            break;
          }
          case "lesson_used": {
            const d = event.data as SSELessonUsedData;
            a.addLessonUsed(d);
            break;
          }
          case "lesson_learned": {
            const d = event.data as SSELessonLearnedData;
            a.setLessonLearned(d);
            break;
          }
          case "done": {
            const d = event.data as SSEDoneData;
            a.finalizeAssistantMessage(d.conversation_id);
            getConversations()
              .then(a.setConversations)
              .catch(() => {});
            break;
          }
          case "warning": {
            const d = event.data as { message?: string };
            if (d.message) {
              import("sonner").then(({ toast }) => toast.warning(d.message));
            }
            break;
          }
          case "error": {
            const d = event.data as SSEErrorData;
            a.setError(d.message);
            break;
          }
        }
      },
      onError: (err: Error) => {
        if (err.name !== "AbortError") {
          actions().setError(err.message);
        }
      },
    });
  }, []);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    actions().setStreaming(false);
  }, []);

  const handleRegenerate = useCallback(() => {
    const msgs = useChatStore.getState().messages;
    const lastUserMsg = [...msgs].reverse().find((m) => m.role === "user");
    if (!lastUserMsg) return;
    actions().removeLastAssistantMessage();
    handleSend(lastUserMsg.content, lastUserMsg.image_base64);
  }, [handleSend]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && useChatStore.getState().streaming) {
        handleStop();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleStop]);

  const lastMsg = messages[messages.length - 1];
  const showRegenerate = lastMsg?.role === "assistant" && !streaming;

  return (
    <div className="flex h-full">
      <ChatSidebar onSelect={loadConversation} />

      <div className="flex flex-1 flex-col min-w-0 min-h-0">
        {/* Messages area */}
        <div ref={scrollContainerRef} className="relative flex-1 overflow-y-auto py-4">
          {messages.length === 0 && !streaming && (
            <div className="flex h-full items-center justify-center">
              <div className="text-center animate-fade-in">
                {/* Nova_ logo */}
                <div className="relative mx-auto mb-6 h-16 w-16">
                  <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-nova-accent/20 to-nova-glow/10 blur-xl" />
                  <div className="relative flex h-16 w-16 items-center justify-center rounded-2xl border border-nova-border bg-nova-surface shadow-[var(--shadow-nova-glow)]">
                    <MessageSquare size={28} strokeWidth={1.5} className="text-nova-accent" />
                  </div>
                </div>
                <p className="text-2xl font-bold bg-gradient-to-r from-nova-text to-nova-glow bg-clip-text text-transparent">Nova_</p>
                <p className="mt-1.5 text-sm text-nova-text-dim">Start a conversation</p>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <ChatMessage key={msg.id ?? `msg-${i}`} message={msg} />
          ))}

          {/* Tool calls during streaming */}
          {toolCalls.map((tc) => (
            <ToolCallCard key={tc.id} toolCall={tc} />
          ))}

          {/* Thinking indicator */}
          {streaming && thinkingStage && (
            <ThinkingIndicator stage={thinkingStage} content={thinkingContent} />
          )}

          {/* Streamed tokens (partial response) */}
          {streamedTokens && (
            <ChatMessage
              message={{ role: "assistant", content: streamedTokens }}
            />
          )}

          {/* Regenerate button */}
          {showRegenerate && (
            <div className="flex justify-start px-4 py-1.5">
              <button
                onClick={handleRegenerate}
                className="flex items-center gap-1.5 rounded-lg border border-nova-border px-3 py-1.5 text-xs text-nova-text-dim hover:bg-nova-surface hover:text-nova-text hover:border-nova-border-bright transition-all duration-200"
              >
                <RotateCcw size={14} />
                Regenerate
              </button>
            </div>
          )}

          {/* Error */}
          {error && (
            <ErrorBanner
              message={error}
              onDismiss={() => actions().setError(null)}
            />
          )}

          {/* Intersection observer sentinel */}
          <div ref={bottomSentinelRef} className="h-1" />
          <div ref={messagesEndRef} />

          {/* Scroll to bottom button */}
          {showScrollBtn && (
            <button
              onClick={() => { userScrolledUpRef.current = false; scrollToBottom(true); }}
              className="sticky bottom-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-1.5 rounded-full border border-nova-border bg-nova-surface/90 backdrop-blur-md px-3.5 py-2 text-xs text-nova-text-dim shadow-[var(--shadow-nova-md)] hover:text-nova-text hover:border-nova-border-bright transition-all duration-200 animate-fade-in"
            >
              <ChevronDown size={14} />
              Scroll to bottom
            </button>
          )}
        </div>

        {/* Input */}
        <ChatInput
          onSend={handleSend}
          onStop={handleStop}
          streaming={streaming}
        />
      </div>
    </div>
  );
}
