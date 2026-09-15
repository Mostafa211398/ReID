import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  globalTeardown: "./tests/global-teardown.ts",
  timeout: 120000,
  workers: 1,
  use: { baseURL: "http://127.0.0.1:8013", channel: "chrome", headless: true, viewport: { width: 1440, height: 1000 }, trace: "retain-on-failure" },
  webServer: {
    command: `"${process.env.IRIS_TEST_PYTHON || '..\\.runtime\\reid-env\\Scripts\\python.exe'}" -m tests.browser_server`,
    env: { PYTHONPATH: "../backend", IRIS_BROWSER_DATA: `../.runtime/browser-${Date.now()}` },
    url: "http://127.0.0.1:8013/api/system/health",
    reuseExistingServer: false,
    timeout: 60000,
    gracefulShutdown: { signal: "SIGINT", timeout: 5000 }
  }
});
