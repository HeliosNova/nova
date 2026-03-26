import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    host: "0.0.0.0",
    allowedHosts: ["nova-frontend", "localhost", ".localhost"],
    watch: {
      usePolling: true,
    },
    proxy: {
      "/api": {
        // Server-side proxy: use internal Docker hostname, not localhost
        target: process.env.API_PROXY_TARGET || "http://nova-app:8000",
        changeOrigin: true,
      },
    },
  },
});
