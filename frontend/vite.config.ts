import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, "..", "IRIS_");
  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: Number(env.IRIS_FRONTEND_PORT || 5176),
      strictPort: true,
      proxy: { "/api": `http://127.0.0.1:${env.IRIS_PORT || 8003}` }
    },
    build: { target: "es2022" }
  };
});
