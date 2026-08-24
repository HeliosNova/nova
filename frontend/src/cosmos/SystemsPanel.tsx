import { lazy, Suspense, useState } from "react";
import { EmbeddedChromeContext } from "../components/ui/PageHeader";

const SettingsPage = lazy(() => import("../pages/SettingsPage"));
const MonitorsPage = lazy(() => import("../pages/MonitorsPage"));
const LearningPage = lazy(() => import("../pages/LearningPage"));
const DocumentsPage = lazy(() => import("../pages/DocumentsPage"));
const ActionsPage = lazy(() => import("../pages/ActionsPage"));

type Sys = "settings" | "monitors" | "learning" | "documents" | "actions";
const TABS: { id: Sys; label: string }[] = [
  { id: "settings", label: "Settings" },
  { id: "monitors", label: "Monitors" },
  { id: "learning", label: "Learning" },
  { id: "documents", label: "Documents" },
  { id: "actions", label: "Audit log" },
];

/** The utility drawer — configuration plus the management surfaces (monitors,
 *  lessons/skills, documents, audit) kept out of the intelligence regions so the
 *  cosmos stays about knowing, not admin. */
export default function SystemsPanel() {
  const [tab, setTab] = useState<Sys>("settings");
  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-1.5">
        {TABS.map((t) => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`rounded-md border px-3 py-1 font-mono text-[10px] uppercase tracking-[0.12em] transition-colors ${tab === t.id ? "border-nova-accent/40 bg-nova-accent/10 text-nova-accent" : "border-nova-border bg-nova-surface/40 text-nova-text-dim hover:text-nova-text"}`}>
            {t.label}
          </button>
        ))}
      </div>
      {/* embedded chrome: the panel header already says where we are — the
          legacy pages drop their own <h1> and keep only their action buttons */}
      <EmbeddedChromeContext.Provider value={true}>
        <Suspense fallback={<div className="h-24 animate-pulse rounded bg-nova-surface/40" />}>
          {tab === "settings" && <SettingsPage />}
          {tab === "monitors" && <MonitorsPage />}
          {tab === "learning" && <LearningPage />}
          {tab === "documents" && <DocumentsPage />}
          {tab === "actions" && <ActionsPage />}
        </Suspense>
      </EmbeddedChromeContext.Provider>
    </div>
  );
}
