import { useEffect, useState, useRef, useCallback } from "react";
import { Pencil } from "lucide-react";
import { useChatStore } from "../lib/store";
import { getConversations, deleteConversation, searchConversations, searchMessages, renameConversation } from "../lib/api";
import { formatDate, truncate } from "../lib/utils";
import { toast } from "sonner";
import type { ConversationSummary, MessageSearchResult } from "../lib/types";

interface Props {
  onSelect: (id: string) => void;
}

type SearchMode = "conversations" | "messages";

export default function ChatSidebar({ onSelect }: Props) {
  const {
    conversations,
    setConversations,
    activeConversationId,
    newChat,
    sidebarOpen,
    toggleSidebar,
  } = useChatStore();
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<ConversationSummary[] | null>(null);
  const [messageResults, setMessageResults] = useState<MessageSearchResult[] | null>(null);
  const [searchMode, setSearchMode] = useState<SearchMode>("conversations");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Rename state
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const renameInputRef = useRef<HTMLInputElement>(null);

  // Mobile detection
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  useEffect(() => {
    const handler = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener("resize", handler);
    return () => window.removeEventListener("resize", handler);
  }, []);

  const [loading, setLoading] = useState(true);

  const refresh = () => {
    getConversations()
      .then((v) => { setConversations(Array.isArray(v) ? v : []); setLoading(false); })
      .catch(() => { setConversations([]); setLoading(false); });
  };

  // Visibility-aware polling — pause when tab is hidden
  useEffect(() => {
    refresh();
    let interval: ReturnType<typeof setInterval>;
    const start = () => { interval = setInterval(refresh, 30_000); };
    const stop = () => clearInterval(interval);
    const onVisibility = () => { document.hidden ? stop() : start(); };
    start();
    document.addEventListener("visibilitychange", onVisibility);
    return () => { stop(); document.removeEventListener("visibilitychange", onVisibility); };
  }, []);

  useEffect(() => {
    if (renamingId && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
    }
  }, [renamingId]);

  const handleSearch = useCallback((q: string) => {
    setSearchQuery(q);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (!q.trim()) {
      setSearchResults(null);
      setMessageResults(null);
      return;
    }
    debounceRef.current = setTimeout(() => {
      if (searchMode === "conversations") {
        searchConversations(q.trim())
          .then((v) => setSearchResults(Array.isArray(v) ? v : []))
          .catch(() => setSearchResults([]));
        setMessageResults(null);
      } else {
        searchMessages(q.trim())
          .then((v) => setMessageResults(Array.isArray(v) ? v : []))
          .catch(() => setMessageResults([]));
        setSearchResults(null);
      }
    }, 300);
  }, [searchMode]);

  // Re-search when mode changes with existing query
  useEffect(() => {
    if (searchQuery.trim()) {
      handleSearch(searchQuery);
    }
  }, [searchMode]);

  const handleDelete = async (id: string) => {
    try {
      await deleteConversation(id);
      setConfirmId(null);
      refresh();
      if (activeConversationId === id) newChat();
    } catch {
      toast.error("Failed to delete conversation");
      setConfirmId(null);
    }
  };

  const handleSelect = (id: string) => {
    onSelect(id);
    if (isMobile) toggleSidebar();
  };

  const startRename = (c: ConversationSummary) => {
    setRenamingId(c.id);
    setRenameValue(c.title || "Untitled");
  };

  const commitRename = async () => {
    if (!renamingId) return;
    const trimmed = renameValue.trim();
    if (trimmed) {
      try {
        await renameConversation(renamingId, trimmed);
        refresh();
      } catch {
        toast.error("Failed to rename conversation");
      }
    }
    setRenamingId(null);
  };

  const cancelRename = () => {
    setRenamingId(null);
  };

  const displayList = searchResults !== null ? searchResults : conversations;

  // Sidebar panel classes
  const panelClasses = isMobile
    ? `fixed left-0 top-0 z-50 h-full w-[280px] pt-0 transition-transform duration-200 ease-in-out ${
        sidebarOpen ? "translate-x-0" : "-translate-x-full"
      }`
    : `transition-all duration-200 ease-in-out ${
        sidebarOpen ? "w-[280px]" : "w-0 overflow-hidden"
      }`;

  return (
    <>
      {/* Backdrop for mobile */}
      {isMobile && sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 animate-fade-in"
          onClick={toggleSidebar}
        />
      )}

      <aside className={`${panelClasses} flex h-full flex-col border-r border-nova-border bg-nova-surface`}>
        <div className="space-y-2 border-b border-nova-border p-3">
          <button
            onClick={() => newChat()}
            className="w-full rounded bg-nova-accent px-3 py-2 text-sm font-medium text-white hover:bg-nova-accent-hover"
          >
            + New Chat
          </button>
          <div className="relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => handleSearch(e.target.value)}
              placeholder={searchMode === "conversations" ? "Search conversations..." : "Search messages..."}
              className="w-full rounded border border-nova-border bg-nova-bg px-3 py-1.5 pl-7 text-xs outline-none focus:border-nova-accent"
            />
            <svg
              className="absolute left-2 top-1/2 -translate-y-1/2 text-nova-text-dim"
              xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
            >
              <circle cx="11" cy="11" r="8"/>
              <line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
            {searchQuery && (
              <button
                onClick={() => handleSearch("")}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-nova-text-dim hover:text-nova-text"
              >
                &times;
              </button>
            )}
          </div>
          {/* Search mode toggle */}
          <div className="flex rounded border border-nova-border text-[10px]">
            <button
              onClick={() => setSearchMode("conversations")}
              className={`flex-1 px-2 py-1 transition-colors ${
                searchMode === "conversations"
                  ? "bg-nova-accent/20 text-nova-accent font-medium"
                  : "text-nova-text-dim hover:text-nova-text"
              }`}
            >
              Conversations
            </button>
            <button
              onClick={() => setSearchMode("messages")}
              className={`flex-1 px-2 py-1 transition-colors ${
                searchMode === "messages"
                  ? "bg-nova-accent/20 text-nova-accent font-medium"
                  : "text-nova-text-dim hover:text-nova-text"
              }`}
            >
              Messages
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          {loading && conversations.length === 0 && (
            <div className="space-y-2 p-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-10 rounded bg-nova-border/30 animate-skeleton-pulse" />
              ))}
            </div>
          )}

          {/* Message search results */}
          {searchMode === "messages" && messageResults !== null && (
            <>
              {messageResults.length === 0 ? (
                <p className="p-4 text-center text-xs text-nova-text-dim">
                  No messages found
                </p>
              ) : (
                messageResults.map((m, i) => (
                  <div
                    key={`msg-${i}`}
                    onClick={() => handleSelect(m.conversation_id)}
                    className="cursor-pointer border-b border-nova-border px-3 py-2.5 text-sm hover:bg-nova-bg"
                  >
                    <div className="flex items-center gap-1.5 mb-0.5">
                      <span className={`rounded px-1 py-0.5 text-[9px] font-medium ${
                        m.role === "user"
                          ? "bg-nova-accent/20 text-nova-accent"
                          : "bg-nova-border text-nova-text-dim"
                      }`}>
                        {m.role}
                      </span>
                      <span className="text-[10px] text-nova-text-dim">{formatDate(m.created_at)}</span>
                    </div>
                    <p className="text-xs text-nova-text-dim line-clamp-2">
                      {m.content.slice(0, 120)}
                    </p>
                  </div>
                ))
              )}
            </>
          )}

          {/* Conversation search results / list */}
          {searchMode === "conversations" && (
            <>
              {searchResults !== null && searchResults.length === 0 && (
                <p className="p-4 text-center text-xs text-nova-text-dim">
                  No results found
                </p>
              )}
              {!loading && displayList.length === 0 && searchResults === null && (
                <p className="p-4 text-center text-xs text-nova-text-dim">
                  No conversations yet
                </p>
              )}
              {displayList.map((c) => (
                <div
                  key={c.id}
                  onClick={() => renamingId !== c.id && handleSelect(c.id)}
                  onDoubleClick={() => startRename(c)}
                  className={`group flex cursor-pointer items-center justify-between border-b border-nova-border px-3 py-2.5 text-sm hover:bg-nova-bg ${
                    activeConversationId === c.id ? "bg-nova-bg" : ""
                  }`}
                >
                  <div className="min-w-0 flex-1">
                    {renamingId === c.id ? (
                      <input
                        ref={renameInputRef}
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        onBlur={commitRename}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") commitRename();
                          if (e.key === "Escape") cancelRename();
                        }}
                        className="w-full rounded border border-nova-accent bg-nova-bg px-1.5 py-0.5 text-sm outline-none"
                        onClick={(e) => e.stopPropagation()}
                      />
                    ) : (
                      <div className="flex items-center gap-1 truncate font-medium">
                        {truncate(c.title || "Untitled", 28)}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            startRename(c);
                          }}
                          className="hidden shrink-0 rounded p-0.5 text-nova-text-dim hover:text-nova-text group-hover:inline-flex"
                          aria-label="Rename conversation"
                          title="Rename"
                        >
                          <Pencil size={11} />
                        </button>
                      </div>
                    )}
                    <div className="text-xs text-nova-text-dim">
                      {formatDate(c.updated_at)}
                    </div>
                  </div>
                  {confirmId === c.id ? (
                    <div className="flex gap-1">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDelete(c.id);
                        }}
                        className="rounded bg-nova-error px-2 py-0.5 text-xs text-white"
                      >
                        Yes
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setConfirmId(null);
                        }}
                        className="rounded bg-nova-border px-2 py-0.5 text-xs"
                      >
                        No
                      </button>
                    </div>
                  ) : (
                    renamingId !== c.id && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setConfirmId(c.id);
                        }}
                        className="hidden text-nova-text-dim hover:text-nova-error group-hover:block"
                        aria-label={`Delete conversation: ${c.title || "Untitled"}`}
                      >
                        &times;
                      </button>
                    )
                  )}
                </div>
              ))}
            </>
          )}
        </div>
      </aside>
    </>
  );
}
