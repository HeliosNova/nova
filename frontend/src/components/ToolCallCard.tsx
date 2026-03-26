import { useState } from "react";
import { CheckCircle, AlertCircle, ChevronDown, ChevronUp } from "lucide-react";
import type { ToolCall } from "../lib/types";

interface Props {
  toolCall: ToolCall;
}

export default function ToolCallCard({ toolCall }: Props) {
  const isExecuting = toolCall.status === "executing";
  const isDelegation = toolCall.tool === "delegate";
  const hasError = toolCall.result?.startsWith("[Tool error") || toolCall.result?.startsWith("[Error");
  const [expanded, setExpanded] = useState(false);

  const isTruncated = toolCall.result && toolCall.result.length > 500;

  return (
    <div className={`mx-4 my-1.5 rounded-lg border px-3.5 py-2.5 text-xs animate-scale-in transition-all duration-300 ${
      isExecuting
        ? isDelegation
          ? "border-nova-accent/30 bg-nova-accent/[0.06] shadow-[0_0_15px_rgba(99,102,241,0.08)]"
          : "border-nova-warning/20 bg-nova-warning/[0.04]"
        : hasError
          ? "border-nova-error/20 bg-nova-error/[0.04]"
          : "border-nova-success/15 bg-nova-success/[0.04]"
    }`}>
      <div className="flex items-center gap-2.5">
        {isExecuting ? (
          /* Progress ring animation */
          <div className="relative h-4 w-4 flex-shrink-0">
            <svg className="h-4 w-4 animate-orb-spin" viewBox="0 0 16 16">
              <circle cx="8" cy="8" r="6" fill="none" strokeWidth="1.5"
                className={isDelegation ? "stroke-nova-accent/20" : "stroke-nova-warning/20"} />
              <circle cx="8" cy="8" r="6" fill="none" strokeWidth="1.5"
                strokeDasharray="12 26" strokeLinecap="round"
                className={isDelegation ? "stroke-nova-accent" : "stroke-nova-warning"} />
            </svg>
          </div>
        ) : hasError ? (
          <AlertCircle size={14} className="text-nova-error flex-shrink-0" />
        ) : (
          <CheckCircle size={14} className="text-nova-success flex-shrink-0" />
        )}
        <span className="font-semibold text-nova-text">{toolCall.tool}</span>
        {toolCall.args && Object.keys(toolCall.args).length > 0 && (
          <span className="text-nova-text-dim truncate max-w-[300px]" title={JSON.stringify(toolCall.args)}>
            {Object.values(toolCall.args).map(v => typeof v === "string" ? v : JSON.stringify(v)).join(", ").slice(0, 80)}
          </span>
        )}
        <span className="ml-auto text-nova-text-dim">
          {isDelegation && isExecuting
            ? "delegating..."
            : isExecuting
            ? "executing..."
            : hasError
            ? "failed"
            : "complete"}
        </span>
      </div>
      {toolCall.result && (
        <div className="mt-1.5">
          <pre className="max-h-48 overflow-auto whitespace-pre-wrap text-nova-text-dim leading-relaxed font-mono text-[11px]">
            {expanded ? toolCall.result : toolCall.result.slice(0, 500)}
            {!expanded && isTruncated && "..."}
          </pre>
          {isTruncated && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="mt-1.5 flex items-center gap-1 text-nova-accent hover:text-nova-accent-hover transition-colors"
            >
              {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              {expanded ? "Show less" : `Show all (${toolCall.result.length} chars)`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
