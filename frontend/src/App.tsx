import { Toaster } from "sonner";
import { useThemeEffect } from "./lib/theme";
import ErrorBoundary from "./components/ErrorBoundary";
import CosmosApp from "./cosmos/CosmosApp";

/**
 * Nova is not a dashboard. The whole app is one continuous space — the knowledge
 * cosmos — and you fly to regions of Nova's mind to open them. No sidebar, no
 * pages, no tabs. (2026-08-21 reconception.)
 */
export default function App() {
  useThemeEffect();
  return (
    <ErrorBoundary>
      <CosmosApp />
      <Toaster richColors position="bottom-right" theme="dark" />
    </ErrorBoundary>
  );
}
