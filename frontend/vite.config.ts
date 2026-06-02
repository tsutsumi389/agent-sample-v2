import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// BACKEND_URL はコンテナ内では http://backend:8000、ローカルでは http://localhost:8000
const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": { target: backendUrl, changeOrigin: true },
    },
  },
});
