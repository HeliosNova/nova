import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

interface Props {
  stage: string | null;
  content?: string | null;
}

export default function ThinkingIndicator({ stage, content }: Props) {
  const [expanded, setExpanded] = useState(false);
  const hasContent = stage === "reasoning" && !!content;

  return (
    <div className="px-4 py-2 animate-fade-in">
      <div className="flex items-center gap-3">
        {/* Orbital thinking animation */}
        <div className="relative flex h-8 w-8 items-center justify-center">
          {/* Outer ring */}
          <span className="absolute inset-0 rounded-full border border-nova-accent/20 animate-orb-spin" />
          {/* Inner spinning dot */}
          <span className="absolute h-full w-full animate-orb-spin">
            <span className="absolute top-0 left-1/2 -translate-x-1/2 h-1.5 w-1.5 rounded-full bg-nova-glow shadow-[0_0_8px_rgba(129,140,248,0.6)]" />
          </span>
          {/* Core orb */}
          <span className="relative h-2.5 w-2.5 rounded-full bg-gradient-to-br from-nova-accent to-nova-glow animate-breathe shadow-[0_0_12px_rgba(99,102,241,0.4)]" />
        </div>

        <div className="flex flex-col">
          {hasContent ? (
            <button
              onClick={() => setExpanded((v) => !v)}
              className="flex items-center gap-1.5 text-xs font-medium text-nova-glow hover:text-nova-accent transition-colors"
            >
              {expanded ? "Hide" : "Show"} reasoning
              {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            </button>
          ) : (
            stage && (
              <span className="text-xs font-medium text-nova-text-dim tracking-wide">
                {stage.replace(/_/g, " ")}
              </span>
            )
          )}
        </div>
      </div>

      {hasContent && expanded && (
        <div className="mt-2 ml-11 max-h-64 overflow-y-auto rounded-lg border border-nova-border bg-nova-bg/80 p-3 text-xs leading-relaxed italic text-nova-text-dim animate-scale-in backdrop-blur-sm">
          {content}
        </div>
      )}
    </div>
  );
}
