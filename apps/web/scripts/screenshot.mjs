// Design QA screenshots (docs/DESIGN.md, "Design QA loop").
// Usage: node scripts/screenshot.mjs <url> <out.png> [width] [height] [dark|light] [waitMs] [cookieHeader]
// Uses software WebGL (SwiftShader) so the 3D avatar renders in headless Chromium.
import { chromium } from "playwright";

const [url, out, w = "1440", h = "900", scheme = "dark", wait = "4000", cookie = ""] = process.argv.slice(2);
if (!url || !out) {
  console.error("usage: node scripts/screenshot.mjs <url> <out.png> [w] [h] [dark|light] [waitMs] [cookie]");
  process.exit(2);
}
const browser = await chromium.launch({
  args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--autoplay-policy=no-user-gesture-required"],
});
const ctx = await browser.newContext({ viewport: { width: +w, height: +h }, colorScheme: scheme, deviceScaleFactor: 1 });
if (cookie) {
  const u = new URL(url);
  await ctx.addCookies(cookie.split(";").map((c) => {
    const [name, ...v] = c.trim().split("=");
    return { name, value: v.join("="), domain: u.hostname, path: "/" };
  }));
}
const page = await ctx.newPage();
const logs = [];
page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") logs.push(`${m.type()}: ${m.text().slice(0, 300)}`); });
page.on("pageerror", (e) => logs.push(`pageerror: ${e.message.slice(0, 300)}`));
await page.goto(url, { waitUntil: "networkidle", timeout: 120_000 }).catch((e) => logs.push(`goto: ${e.message}`));
await page.waitForTimeout(+wait);
await page.screenshot({ path: out, fullPage: process.env.FULL === "1", timeout: 180_000 });
console.log(logs.length ? logs.join("\n") : "no console errors");
await browser.close();
