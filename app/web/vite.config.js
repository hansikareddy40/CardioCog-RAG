import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the screen runs on port 5173 and forwards /api to the Python API.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://localhost:8000" } },
});
