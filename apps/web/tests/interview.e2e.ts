// End-to-end: a spoken interview through the real pipeline.
// Browser mic (fake device fed with recorded speech) -> AudioWorklet -> WebSocket -> server VAD +
// speech recognition (sherpa-onnx) -> interviewer agent (LLM) -> Kokoro TTS -> 3D avatar playback.
//
// E2E_REQUIRE_LLM=1 (CI "e2e-full" job with a real LLM): every question must come from the LLM; a
// single flagged backup question fails the test. Without it (no LLM configured), the test instead
// checks that backup questions are clearly flagged in the UI.
import { expect, test, type Page } from "@playwright/test";
import { blockMedia, captureConsole, installFakeMedia, newSession, speak } from "./helpers";

const REQUIRE_LLM = process.env.E2E_REQUIRE_LLM === "1";
// A CPU-hosted 7B model (CI) can take minutes per call; hosted APIs answer in seconds.
const LLM_TIMEOUT = REQUIRE_LLM ? 600_000 : 60_000;

// On failure, print what the browser logged and the room's protocol trace (CI artifacts are not
// always reachable, the job log is).
let consoleLines: string[] = [];
test.beforeEach(({ page }) => { consoleLines = captureConsole(page); });
test.afterEach(async ({ page }, info) => {
  if (info.status === info.expectedStatus) return;
  // a frozen page must not eat the rest of the timeout: give the diagnostics 5 s
  const events = await Promise.race([
    page.evaluate(() => (window as unknown as { __room?: { diag: { events: string[] } } }).__room?.diag.events ?? []).catch(() => [] as string[]),
    new Promise<string[]>((r) => setTimeout(() => r(["(page did not respond)"]), 5000)),
  ]);
  console.log(`[${info.project.name}] FAILED "${info.title}"\nprotocol events:\n${events.join("\n")}\nbrowser console:\n${consoleLines.slice(-60).join("\n")}`);
});

async function interviewerLines(page: Page) {
  return page.getByTestId("transcript").locator("li").filter({ hasNotText: /^You/ }).count();
}

async function phase(page: Page, name: "Listening" | "Speaking" | "Thinking", timeout: number) {
  await expect(page.getByRole("banner").getByText(name, { exact: true })).toBeVisible({ timeout });
}

test("spoken interview: speech in, live captions, LLM follow-ups, spoken questions, barge-in, report", async ({ page }, info) => {
  test.setTimeout(REQUIRE_LLM ? 1_500_000 : 600_000);
  await installFakeMedia(page);
  await page.goto("/");
  const id = await newSession(page, {
    role: "Backend Engineer", seniority: "senior", company: "Acme Payments", interview_type: "technical",
    skills: "Postgres, API design", duration_minutes: "5",
    job_description: "Own the billing platform: Postgres, batch jobs, REST APIs, on-call.",
  });
  await page.goto(`/interview/${id}?debug=1`);

  // Pre-join device check
  await expect(page.getByRole("heading", { name: "Check your camera and microphone" })).toBeVisible({ timeout: 90_000 });
  const join = page.getByTestId("join");
  await expect(join).toBeEnabled({ timeout: 240_000 }); // interviewer (3D avatar or audio fallback) ready
  await page.screenshot({ path: info.outputPath("01-device-check.png") });
  await join.click();

  // First question is spoken, then the room listens
  await expect(page.getByTestId("transcript").locator("li").first()).toBeVisible({ timeout: LLM_TIMEOUT });
  await phase(page, "Listening", 120_000);

  // Answer 1 (US English voice): live captions while speaking, final transcript afterwards
  const d1 = await speak(page, "answer-1-us.wav");
  await expect(page.getByTestId("captions")).toContainText(/billing|nightly|batch/i, { timeout: 20_000 });

  // Next question, generated after the answer. Barge-in: talk over it while its audio is playing.
  // Watch the room's state rather than the DOM: on a busy machine the page can re-render seconds
  // after the audio has started.
  await page.waitForFunction(() => {
    const room = (window as unknown as { __room?: { getSnapshot(): { audioPlaying: boolean; transcript: { role: string }[] } } }).__room;
    const s = room?.getSnapshot();
    return !!s && s.audioPlaying && s.transcript.filter((t) => t.role === "interviewer").length >= 2;
  }, null, { timeout: LLM_TIMEOUT + d1 * 1000, polling: 50 });
  const d2 = await speak(page, "answer-2-in.wav"); // Indian-English voice, started while the question plays
  await expect(page.getByTestId("transcript")).toContainText(/(forty|40) minutes/i, { timeout: 60_000 });
  expect(await interviewerLines(page)).toBeGreaterThanOrEqual(2);
  await phase(page, "Listening", 30_000);
  await page.screenshot({ path: info.outputPath("03-barge-in.png") });
  await page.waitForTimeout(d2 * 1000);
  await expect(page.getByTestId("transcript")).toContainText(/design review|trade.?offs|versioned/i, { timeout: 60_000 });
  const events = await page.evaluate(() => (window as unknown as { __room?: { diag: { events: string[] } } }).__room?.diag.events ?? []);
  await info.attach("protocol-events", { body: events.join("\n"), contentType: "text/plain" });
  console.log(`[${info.project.name}] protocol events:\n${events.join("\n")}`);
  await expect(page.getByTestId("diag")).toContainText(/barge-ins\s*[1-9]/);
  if (info.project.name === "chromium") {
    // the 3D avatar's mouth is driven by the interviewer audio that actually played
    await expect(page.getByTestId("diag")).toContainText(/talkinghead.*visemes [1-9]/);
  }

  if (REQUIRE_LLM) {
    await expect.poll(() => interviewerLines(page), { timeout: LLM_TIMEOUT }).toBeGreaterThanOrEqual(3);
    await expect(page.getByTestId("diag")).toContainText(/emergency questions\s*0/);
    await expect(page.getByText("Backup question (AI unavailable)")).toHaveCount(0);
  } else {
    await expect(page.getByText("Backup question (AI unavailable)").first()).toBeVisible();
  }
  await page.screenshot({ path: info.outputPath("04-transcript.png") });

  // End early -> report
  await page.getByTestId("end").click();
  await page.getByRole("button", { name: "End and see report" }).click();
  await page.waitForURL(new RegExp(`/reports/${id}`), { timeout: 180_000 });
  await expect(page.getByText("Interview report", { exact: true })).toBeVisible({ timeout: 120_000 });
  await expect(page.getByRole("heading", { level: 1, name: "Backend Engineer · Acme Payments" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Answer by answer" })).toBeVisible();
  await page.screenshot({ path: info.outputPath("05-report.png"), fullPage: true });
});

test("microphone blocked: clear recovery steps and typing still works", async ({ page }) => {
  await blockMedia(page);
  await page.goto("/");
  const id = await newSession(page, { role: "Product Manager", interview_type: "behavioral", duration_minutes: "5" });
  await page.goto(`/interview/${id}`);
  // Engines word a blocked mic differently (denied / insecure / unsupported); each must show recovery steps.
  const help = page.getByRole("status").filter({ hasText: /Microphone access is blocked|isn't secure|can't capture audio|Couldn't start the microphone/ });
  await expect(help.first()).toBeVisible({ timeout: 90_000 });
  console.log(`[${test.info().project.name}] mic error shown: ${(await help.first().innerText()).split("\n")[0]}`);
  const typeInstead = page.getByRole("button", { name: "Type answers instead" });
  await expect(typeInstead).toBeEnabled({ timeout: 240_000 });
  await typeInstead.click();
  await expect(page.getByTestId("transcript").locator("li").first()).toBeVisible({ timeout: LLM_TIMEOUT });
  await page.getByLabel("Type your answer").fill("I led a pricing experiment that lifted conversion by four percent; I wrote the hypothesis and ran the A/B test.");
  await page.getByTestId("send-text").click();
  await expect(page.getByTestId("transcript")).toContainText("pricing experiment");
  await expect.poll(() => interviewerLines(page), { timeout: LLM_TIMEOUT }).toBeGreaterThanOrEqual(2);
});

test("API rejects state changes without the CSRF header and unauthenticated access", async ({ request }) => {
  expect((await request.post("/api/auth/logout")).status()).toBe(403);
  expect((await request.get("/api/sessions")).status()).toBe(401);
});
