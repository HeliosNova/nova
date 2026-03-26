import { cn } from "@/lib/utils";

interface Tab {
  id: string;
  label: string;
  badge?: boolean;
}

interface Props {
  tabs: Tab[];
  active: string;
  onChange: (id: string) => void;
}

export default function Tabs({ tabs, active, onChange }: Props) {
  return (
    <div className="mb-4 flex gap-1 border-b border-nova-border pb-2 overflow-x-auto scrollbar-hide">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={cn(
            "relative rounded-t px-3 py-1.5 text-sm transition-colors",
            active === tab.id
              ? "text-nova-accent font-medium"
              : "text-nova-text-dim hover:text-nova-text",
          )}
        >
          <span className="flex items-center gap-1.5">
            {tab.label}
            {tab.badge && (
              <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-nova-success" title="Auto-learning enabled" />
            )}
          </span>
          {active === tab.id && (
            <span className="absolute bottom-0 left-1 right-1 h-0.5 rounded-full bg-gradient-to-r from-nova-accent to-nova-glow" />
          )}
        </button>
      ))}
    </div>
  );
}
