import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// The API base URL is read at runtime from VITE_API_BASE_URL (see .env.example).
// A dev proxy is offered as a fallback so `fetch("/api/...")` also works.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      "/api": {
        target: process.env.VITE_API_BASE_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  preview: { port: 4173 },
  build: { outDir: "dist", sourcemap: false },
});
