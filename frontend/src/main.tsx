import React from "react";
import ReactDOM from "react-dom/client";
// Type system, all self-hosted (no CDN — sovereign/offline-safe), 2026-08-21:
//   Inter          — body / reading (dense data reads well)
//   Space Grotesk  — display / headings (technical-geometric, has a POV)
//   JetBrains Mono — DATA: timestamps, confidences, counts = instrument readings.
// The mono-for-data split is the structural device: in an instrument, numbers
// are readings, not prose. Before this everything fell back to Segoe UI.
import "@fontsource-variable/inter";
import "@fontsource-variable/space-grotesk";
import "@fontsource-variable/jetbrains-mono";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
