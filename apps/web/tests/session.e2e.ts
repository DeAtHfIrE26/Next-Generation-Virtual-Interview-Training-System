import { expect, test } from "@playwright/test";

test("candidate signs up, consents, completes an interview and gets a shareable report", async ({ page }) => {
  const email = `e2e-${Date.now()}@example.com`;
  await page.goto("/signup");
  await page.getByLabel("Name").fill("E2E Candidate");
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Password (10+ characters)").fill("correct horse battery");
  await page.getByRole("checkbox", { name: "I am 18 or older." }).check();
  await page.getByRole("checkbox", { name: /I agree to the/ }).check();
  await page.getByRole("button", { name: "Create account" }).click();

  await expect(page.getByRole("heading", { name: "Before you start" })).toBeVisible();
  await page.getByRole("checkbox", { name: /Run practice interviews/ }).check();
  await expect(page.getByRole("button", { name: "Continue" })).toBeEnabled();
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByRole("heading", { name: /^Hi/ })).toBeVisible();

  await page.goto("/interview/new");
  await page.getByLabel("Target role").fill("Backend Software Engineer");
  await page.getByLabel("Length").selectOption("5");
  await page.getByRole("button", { name: "Start interview" }).click();

  await page.getByRole("button", { name: "Start the interview" }).click();
  for (let i = 0; i < 12; i++) {
    const done = page.getByRole("button", { name: "Done answering" });
    const skip = page.getByRole("button", { name: "Skip to answering" });
    await expect(done.or(skip).or(page.getByRole("heading", { name: "Interview report" }))).toBeVisible({ timeout: 60_000 });
    if (await page.getByRole("heading", { name: "Interview report" }).isVisible()) break;
    if (await skip.isVisible()) await skip.click();
    await page.getByLabel(/type your answer/i).fill(
      "At my previous job I led the migration of our billing service. I profiled the slow queries, added indexes " +
      "and rewrote the batch job. As a result the nightly run dropped from six hours to forty minutes.");
    await page.getByRole("button", { name: "Done answering" }).click();
    const next = page.getByRole("button", { name: "Next question" });
    await expect(next).toBeVisible();
    await expect(page.getByText("experimental").first()).toBeVisible();
    await next.click();
  }

  await expect(page.getByRole("heading", { name: "Interview report" })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText(/Q1/).first()).toBeVisible();
  await page.getByRole("button", { name: "Create share link" }).click();
  const link = (await page.locator("code").innerText()).trim();
  expect(link).toContain("/share/");

  const shared = await page.context().newPage();
  await shared.goto(link.replace(/^https?:\/\/[^/]+/, ""));
  await expect(shared.getByRole("heading", { name: "Interview report" })).toBeVisible();

  await page.goto("/dashboard");
  await expect(page.getByRole("link", { name: "Report" })).toBeVisible();
});

test("API rejects state changes without the CSRF header and unauthenticated access", async ({ request }) => {
  expect((await request.post("/api/auth/logout")).status()).toBe(403);
  expect((await request.get("/api/sessions")).status()).toBe(401);
});
