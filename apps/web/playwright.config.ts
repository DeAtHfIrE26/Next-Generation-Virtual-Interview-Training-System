import { defineConfig, devices } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const apiPort = 8100, webPort = 3100;
const e2eDb = path.join(repo, "services/api/var/e2e.db");

export default defineConfig({
  testDir: "tests",
  testMatch: "**/*.e2e.ts",
  timeout: 180_000,
  expect: { timeout: 20_000 },
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${webPort}`,
    ...devices["Desktop Chrome"],
    permissions: ["camera", "microphone"],
    launchOptions: {
      args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"],
      ...(process.env.PW_CHROMIUM_PATH ? { executablePath: process.env.PW_CHROMIUM_PATH } : {}),
    },
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: `rm -f ${e2eDb} && uv run uvicorn interview_api.main:app --port ${apiPort}`,
      cwd: repo,
      url: `http://127.0.0.1:${apiPort}/health`,
      reuseExistingServer: false,
      env: { DATABASE_URL: `sqlite:///${e2eDb}`, TEMPLATE_KEK_BASE64: "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s=",
             LLM_PROVIDER: "none", APP_BASE_URL: `http://127.0.0.1:${webPort}` },
      timeout: 120_000,
    },
    {
      command: `npx next start -p ${webPort}`,
      url: `http://127.0.0.1:${webPort}`,
      reuseExistingServer: false,
      env: { API_BASE_URL: `http://127.0.0.1:${apiPort}` },
      timeout: 120_000,
    },
  ],
});
