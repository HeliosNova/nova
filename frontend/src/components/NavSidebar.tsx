import { useState, useEffect } from "react";
import {
  MessageSquare,
  GraduationCap,
  FileText,
  Activity,
  Zap,
  Settings,
  PanelLeftClose,
  PanelLeftOpen,
  Menu,
  X,
} from "lucide-react";
import type { Tab } from "@/App";
import { StatusBadge } from "@/components/ui";
import { useSettingsStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { Sun, Moon, Monitor } from "lucide-react";

const navItems: { id: Tab; label: string; icon: typeof MessageSquare }[] = [
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "learning", label: "Learning", icon: GraduationCap },
  { id: "documents", label: "Documents", icon: FileText },
  { id: "monitors", label: "Monitors", icon: Activity },
  { id: "actions", label: "Actions", icon: Zap },
  { id: "settings", label: "Settings", icon: Settings },
];

interface Props {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
}

export default function NavSidebar({ activeTab, onTabChange }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const { theme, setTheme } = useSettingsStore();

  useEffect(() => {
    const handler = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener("resize", handler);
    return () => window.removeEventListener("resize", handler);
  }, []);

  const handleNav = (tab: Tab) => {
    onTabChange(tab);
    if (isMobile) setMobileOpen(false);
  };

  const nextTheme = () => {
    const order: ("dark" | "light" | "system")[] = ["dark", "light", "system"];
    const idx = order.indexOf(theme);
    setTheme(order[(idx + 1) % order.length]);
  };

  const ThemeIcon = theme === "light" ? Sun : theme === "system" ? Monitor : Moon;

  // Desktop sidebar
  const sidebar = (
    <aside
      className={cn(
        "flex h-full flex-col border-r border-nova-border bg-nova-surface/60 backdrop-blur-xl transition-all duration-300",
        isMobile ? "w-[220px]" : expanded ? "w-[220px]" : "w-14",
      )}
    >
      {/* Logo */}
      <div className="flex h-13 items-center border-b border-nova-border px-3">
        {(expanded || isMobile) ? (
          <div className="flex w-full items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-nova-accent animate-breathe shadow-[0_0_8px_rgba(99,102,241,0.4)]" />
              <span className="text-lg font-bold tracking-tight bg-gradient-to-r from-nova-text to-nova-glow bg-clip-text text-transparent">Nova</span>
            </div>
            {isMobile ? (
              <button onClick={() => setMobileOpen(false)} className="rounded-lg p-1.5 text-nova-text-dim hover:text-nova-text hover:bg-nova-border/50 transition-colors">
                <X size={18} />
              </button>
            ) : (
              <button onClick={() => setExpanded(false)} className="rounded-lg p-1.5 text-nova-text-dim hover:text-nova-text hover:bg-nova-border/50 transition-colors">
                <PanelLeftClose size={18} />
              </button>
            )}
          </div>
        ) : (
          <button
            onClick={() => setExpanded(true)}
            className="mx-auto rounded-lg p-1.5 text-nova-text-dim hover:text-nova-text hover:bg-nova-border/50 transition-colors"
            aria-label="Expand sidebar"
          >
            <PanelLeftOpen size={18} />
          </button>
        )}
      </div>

      {/* Nav items */}
      <nav className="flex-1 space-y-0.5 px-2 py-3" aria-label="Main navigation">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => handleNav(item.id)}
              className={cn(
                "relative flex w-full items-center gap-3 rounded-lg px-2.5 py-2 text-sm transition-all duration-200",
                isActive
                  ? "bg-nova-accent/10 text-nova-accent font-medium shadow-[var(--shadow-nova-glow)]"
                  : "text-nova-text-dim hover:text-nova-text hover:bg-nova-border/40",
              )}
              title={!expanded && !isMobile ? item.label : undefined}
            >
              {/* Active indicator bar */}
              {isActive && (
                <span className="absolute left-0 top-1/2 -translate-y-1/2 h-5 w-[3px] rounded-r-full bg-nova-accent shadow-[0_0_8px_rgba(99,102,241,0.5)]" />
              )}
              <Icon size={18} />
              {(expanded || isMobile) && <span>{item.label}</span>}
            </button>
          );
        })}
      </nav>

      {/* Bottom: health + theme toggle */}
      <div className="border-t border-nova-border px-2 py-3 space-y-2">
        <button
          onClick={nextTheme}
          className={cn(
            "flex w-full items-center gap-3 rounded-lg px-2.5 py-2 text-sm text-nova-text-dim hover:text-nova-text hover:bg-nova-border/40 transition-all duration-200",
          )}
          title={`Theme: ${theme}`}
        >
          <ThemeIcon size={18} />
          {(expanded || isMobile) && <span className="capitalize">{theme}</span>}
        </button>
        <div className={cn("flex items-center px-2.5", expanded || isMobile ? "gap-3" : "justify-center")}>
          <StatusBadge />
        </div>
      </div>
    </aside>
  );

  if (isMobile) {
    return (
      <>
        {/* Mobile hamburger */}
        <button
          onClick={() => setMobileOpen(true)}
          className="fixed left-3 top-3 z-40 rounded-xl border border-nova-border bg-nova-surface/80 backdrop-blur-md p-2.5 text-nova-text-dim hover:text-nova-text shadow-[var(--shadow-nova-md)] md:hidden transition-colors"
          aria-label="Open navigation"
        >
          <Menu size={20} />
        </button>

        {/* Overlay */}
        {mobileOpen && (
          <div
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm animate-fade-in md:hidden"
            onClick={() => setMobileOpen(false)}
          />
        )}

        {/* Drawer */}
        <div
          className={cn(
            "fixed left-0 top-0 z-50 h-full transition-transform duration-300 md:hidden",
            mobileOpen ? "translate-x-0" : "-translate-x-full",
          )}
        >
          {sidebar}
        </div>
      </>
    );
  }

  return sidebar;
}
