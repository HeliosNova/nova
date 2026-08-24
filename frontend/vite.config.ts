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
  build: {
    rollupOptions: {
      output: {
        // Split the heavyweights out of the entry chunk (was a 1.3MB index.js):
        // the 3D stack loads in parallel with the app shell and caches
        // independently of app-code changes.
        manualChunks: {
          three: ["three", "@react-three/fiber", "@react-three/drei", "@react-three/postprocessing", "postprocessing"],
          markdown: ["react-markdown", "remark-gfm"],
        },
      },
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
