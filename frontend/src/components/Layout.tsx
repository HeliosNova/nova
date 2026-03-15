import type { ReactNode } from "react";
import type { Tab } from "@/App";
import NavSidebar from "./NavSidebar";

interface Props {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
  children: ReactNode;
}

export default function Layout({ activeTab, onTabChange, children }: Props) {
  return (
    <div className="flex h-screen bg-nova-bg">
      <NavSidebar activeTab={activeTab} onTabChange={onTabChange} />
      <main className="flex-1 flex flex-col min-w-0 min-h-0 relative">
        {/* Subtle ambient glow behind content */}
        <div className="pointer-events-none absolute inset-0 overflow-hidden">
          <div className="absolute -top-32 left-1/2 -translate-x-1/2 h-64 w-96 rounded-full bg-nova-accent/[0.03] blur-3xl" />
        </div>
        <div className="relative flex-1 flex flex-col min-w-0 min-h-0">{children}</div>
      </main>
    </div>
  );
}
