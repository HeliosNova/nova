import { useState, useEffect, lazy, Suspense } from "react";
import { Toaster } from "sonner";
import { useThemeEffect } from "./lib/theme";
import Layout from "./components/Layout";
import ErrorBoundary from "./components/ErrorBoundary";

// Eagerly load chat (most common page), lazy-load the rest
import ChatPage from "./pages/ChatPage";
const LearningPage = lazy(() => import("./pages/LearningPage"));
const DocumentsPage = lazy(() => import("./pages/DocumentsPage"));
const MonitorsPage = lazy(() => import("./pages/MonitorsPage"));
const ActionsPage = lazy(() => import("./pages/ActionsPage"));
const SettingsPage = lazy(() => import("./pages/SettingsPage"));

export type Tab = "chat" | "learning" | "documents" | "monitors" | "actions" | "settings";

function PageFallback() {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="h-6 w-6 animate-spin rounded-full border-2 border-nova-accent border-t-transparent" />
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("chat");

  useThemeEffect();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === "n") {
        e.preventDefault();
        setActiveTab("chat");
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  return (
    <ErrorBoundary>
      <Layout activeTab={activeTab} onTabChange={setActiveTab}>
        {activeTab === "chat" && <ChatPage />}
        <Suspense fallback={<PageFallback />}>
          {activeTab === "learning" && <LearningPage />}
          {activeTab === "documents" && <DocumentsPage />}
          {activeTab === "monitors" && <MonitorsPage />}
          {activeTab === "actions" && <ActionsPage />}
          {activeTab === "settings" && <SettingsPage />}
        </Suspense>
      </Layout>
      <Toaster richColors position="bottom-right" />
    </ErrorBoundary>
  );
}
