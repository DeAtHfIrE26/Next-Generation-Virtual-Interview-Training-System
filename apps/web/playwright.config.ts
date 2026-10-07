import { defineConfig, devices } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

// E2E against the real stack: FastAPI with the local speech models (MODELS_DIR) and whatever LLM
// the environment configures (LLM_PROVIDER etc. are passed through), and the production web build.
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const apiPort = 8100, webPort = 3100;
const e2eDb = path.join(repo, "services/api/var/e2e.db");
const passthrough = Object.fromEntries(
  ["LLM_PROVIDER", "LLM_MODEL", "LLM_FALLBACK_PROVIDER", "OPENAI_COMPAT_BASE_URL", "OPENAI_COMPAT_MODEL", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MODELS_DIR", "STT_THREADS", "TTS_THREADS"]
    .filter((k) => process.env[k]).map((k) => [k, process.env[k] as string]),
);
const chromiumArgs = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--autoplay-policy=no-user-gesture-required"];

export default defineConfig({
  testDir: "tests",
  testMatch: "**/*.e2e.ts",
  timeout: 600_000,
  expect: { timeout: 20_000 },
  retries: 0,
  workers: 1,
  reporter: [["list"], ["html", { open: "never", outputFolder: "playwright-report" }]],
  use: {
    // E2E_BASE_URL targets an already-running stack (e.g. `docker compose up`) instead of starting one.
    baseURL: process.env.E2E_BASE_URL ?? `http://127.0.0.1:${webPort}`,
    trace: "retain-on-failure",
    video: process.env.E2E_VIDEO === "1" ? "on" : "retain-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 }, launchOptions: { args: chromiumArgs, ...(process.env.PW_CHROMIUM_PATH ? { executablePath: process.env.PW_CHROMIUM_PATH } : {}) } } },
    { name: "edge", use: { ...devices["Desktop Edge"], channel: "msedge", viewport: { width: 1440, height: 900 }, launchOptions: { args: chromiumArgs } } },
    { name: "firefox", use: { ...devices["Desktop Firefox"], viewport: { width: 1440, height: 900 }, launchOptions: { firefoxUserPrefs: { "media.autoplay.default": 0, "media.autoplay.blocking_policy": 0 } } } },
    { name: "webkit", use: { ...devices["Desktop Safari"], viewport: { width: 1440, height: 900 } } },
    { name: "mobile-chrome", use: { ...devices["Pixel 7"], launchOptions: { args: chromiumArgs } } },
    { name: "mobile-safari", use: { ...devices["iPhone 14"] } },
  ],
  webServer: process.env.E2E_BASE_URL ? undefined : [
    {
      command: `rm -f ${e2eDb} && uv run uvicorn interview_api.main:app --port ${apiPort}`,
      cwd: repo,
      url: `http://127.0.0.1:${apiPort}/health`,
      reuseExistingServer: !!process.env.E2E_REUSE,
      env: {
        DATABASE_URL: `sqlite:///${e2eDb}`, TEMPLATE_KEK_BASE64: "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s=",
        LLM_PROVIDER: "none", APP_BASE_URL: `http://127.0.0.1:${webPort}`, PUBLIC_API_URL: `http://127.0.0.1:${apiPort}`,
        RATE_LIMIT_MULTIPLIER: "50", ...passthrough,
      },
      timeout: 300_000,
    },
    {
      command: `npx next start -p ${webPort}`,
      url: `http://127.0.0.1:${webPort}`,
      reuseExistingServer: !!process.env.E2E_REUSE,
      env: { API_BASE_URL: `http://127.0.0.1:${apiPort}`, PUBLIC_API_URL: `http://127.0.0.1:${apiPort}` },
      timeout: 120_000,
    },
  ],
});
