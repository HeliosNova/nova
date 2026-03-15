import React, { useState, useCallback, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Check, Copy, BookOpen, ChevronUp, ChevronDown, ExternalLink } from "lucide-react";
import type { Message } from "../lib/types";
import { formatDate } from "../lib/utils";

// Stable references to avoid re-creating on every render
const remarkPluginsList = [remarkGfm];

interface Props {
  message: Message;
}

function CopyButton({ text, className }: { text: string; className?: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* ignore */ }
  }, [text]);

  return (
    <button
      onClick={handleCopy}
      className={`rounded-md p-1.5 text-nova-text-dim hover:bg-nova-border-bright/50 hover:text-nova-text transition-all duration-200 ${className ?? ""}`}
      title="Copy"
    >
      {copied ? <Check size={14} className="text-nova-success" /> : <Copy size={14} />}
    </button>
  );
}

const markdownComponents = {
  pre({ children, ...props }: React.ComponentPropsWithoutRef<"pre">) {
    const codeEl = (children as React.ReactElement[])?.[0];
    const codeText =
      typeof codeEl === "object" && codeEl?.props?.children
        ? String(codeEl.props.children)
        : "";
    return (
      <div className="relative group/code">
        <pre {...props}>{children}</pre>
        <CopyButton
          text={codeText}
          className="absolute right-2 top-2 opacity-0 group-hover/code:opacity-100"
        />
      </div>
    );
  },
};

export default React.memo(function ChatMessage({ message }: Props) {
  const isUser = message.role === "user";
  const [showLessons, setShowLessons] = useState(false);
  const [imageExpanded, setImageExpanded] = useState(false);

  const imageSrc = useMemo(() => {
    if (!message.image_base64) return "";
    if (message.image_base64.startsWith("data:")) return message.image_base64;
    return `data:image/jpeg;base64,${message.image_base64}`;
  }, [message.image_base64]);

  return (
    <div className={`group relative flex ${isUser ? "justify-end" : "justify-start"} px-4 py-1.5 animate-fade-in`}>
      <div
        className={`max-w-[75%] rounded-xl px-4 py-3 text-sm leading-relaxed transition-shadow duration-300 ${
          isUser
            ? "bg-gradient-to-br from-nova-accent to-indigo-600 text-white shadow-[var(--shadow-nova-md)]"
            : "bg-nova-surface border-l-2 border-nova-accent/20 border border-l-nova-accent/30 border-nova-border text-nova-text shadow-[var(--shadow-nova-sm)] hover:shadow-[var(--shadow-nova-md)]"
        }`}
      >
        {/* User image preview */}
        {isUser && message.image_base64 && (
          <>
            <img
              src={imageSrc}
              alt="Uploaded"
              onClick={() => setImageExpanded(true)}
              className="mb-2.5 max-h-36 cursor-pointer rounded-lg border border-white/20 object-cover hover:opacity-80 transition-opacity shadow-[var(--shadow-nova-sm)]"
            />
            {imageExpanded && (
              <div
                className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm animate-fade-in"
                onClick={() => setImageExpanded(false)}
              >
                <img
                  src={imageSrc}
                  alt="Expanded"
                  className="max-h-[85vh] max-w-[85vw] rounded-lg shadow-[var(--shadow-nova-lg)]"
                />
              </div>
            )}
          </>
        )}

        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="prose prose-sm max-w-none prose-p:my-1.5 prose-pre:bg-nova-bg prose-pre:border prose-pre:border-nova-border prose-pre:rounded-lg prose-code:text-nova-glow prose-a:text-nova-accent prose-a:no-underline hover:prose-a:underline prose-headings:text-nova-text prose-p:text-nova-text prose-li:text-nova-text prose-strong:text-nova-text prose-td:text-nova-text prose-th:text-nova-text prose-blockquote:text-nova-text-dim prose-blockquote:border-nova-accent/30">
            <ReactMarkdown
              remarkPlugins={remarkPluginsList}
              components={markdownComponents}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {/* Lesson used badge */}
        {!isUser && message.lessons_used && message.lessons_used.length > 0 && (
          <div className="mt-2.5 border-t border-nova-border/30 pt-2">
            <button
              onClick={() => setShowLessons(!showLessons)}
              className="flex items-center gap-1.5 rounded-full bg-nova-accent/[0.08] px-2.5 py-1 text-[11px] font-medium text-nova-accent/80 hover:text-nova-accent hover:bg-nova-accent/[0.12] transition-all"
            >
              <BookOpen size={11} />
              {message.lessons_used.length} lesson{message.lessons_used.length > 1 ? "s" : ""} applied
              {showLessons ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
            </button>
            {showLessons && (
              <div className="mt-1.5 space-y-1 animate-scale-in">
                {message.lessons_used.map((l) => (
                  <div key={l.lesson_id} className="text-[11px] text-nova-text-dim pl-2 border-l border-nova-accent/20">
                    {l.topic} ({Math.round(l.confidence * 100)}%)
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Lesson learned confirmation */}
        {!isUser && message.lesson_learned && (
          <div className="mt-2.5 flex items-center gap-1.5 rounded-lg bg-nova-success/[0.08] border border-nova-success/15 px-2.5 py-1.5 text-[11px] font-medium text-nova-success animate-scale-in">
            <Check size={12} />
            Got it — I'll remember that about "{message.lesson_learned.topic}"
          </div>
        )}

        {/* Sources */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-2.5 border-t border-nova-border/30 pt-2">
            <div className="flex flex-wrap gap-1.5">
              {message.sources.map((src, i) => {
                const title = String(src.title || src.source || src.name || `Source ${i + 1}`);
                const url = src.url ? String(src.url) : null;
                return url ? (
                  <a
                    key={i}
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 rounded-full bg-nova-accent/[0.08] px-2 py-0.5 text-[10px] font-medium text-nova-accent/80 hover:text-nova-accent hover:bg-nova-accent/[0.12] transition-all"
                  >
                    <ExternalLink size={9} />
                    {title.slice(0, 40)}
                  </a>
                ) : (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1 rounded-full bg-nova-border/40 px-2 py-0.5 text-[10px] font-medium text-nova-text-dim"
                  >
                    {title.slice(0, 40)}
                  </span>
                );
              })}
            </div>
          </div>
        )}

        {/* Timestamp */}
        {message.created_at && (
          <div
            className="mt-1.5 text-[10px] text-nova-text-dim/40 tracking-wide"
            title={message.created_at}
          >
            {formatDate(message.created_at)}
          </div>
        )}
      </div>

      {/* Copy button */}
      <div className={`${isUser ? "mr-1" : "ml-1"} mt-1 opacity-0 group-hover:opacity-100 transition-all duration-200`}>
        <CopyButton text={message.content} />
      </div>
    </div>
  );
});
