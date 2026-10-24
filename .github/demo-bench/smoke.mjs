// Smoke test of the landing-page demo: headless Chromium with a fake
// microphone that plays a public clip; hold the talk button, release, and
// expect the clip's words in the box.
//
// usage: node smoke.mjs <page url> <clip.wav> <word> [<word>...]
import { chromium } from "playwright";

const [url, wav, ...words] = process.argv.slice(2);
const browser = await chromium.launch({
  headless: true,
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-audio-capture=${wav}`],
});
const page = await (await browser.newContext({ permissions: ["microphone"] })).newPage();
const logs = [];
page.on("console", (m) => logs.push(m.type() + ": " + m.text()));
page.on("pageerror", (e) => logs.push("pageerror: " + e.message));
let ok = false;
try {
  await page.goto(url);
  const talk = page.locator("#talk");
  await talk.hover();  // the demo starts downloading on hover
  await page.waitForFunction(() => document.getElementById("try-status").textContent.startsWith("ready"), null, { timeout: 300_000 });
  const fellBack = logs.some((l) => l.includes("falling back"));
  console.log("model ready:", new URL(url).searchParams.get("engine") || "default", fellBack ? "(FELL BACK)" : "");
  await talk.hover();
  await page.mouse.down();
  await page.waitForTimeout(800);
  const preview = await page.locator("#try-out .pre").textContent({ timeout: 1000 }).catch(() => "");
  await page.waitForTimeout(7000);
  console.log("preview during recording:", JSON.stringify(await page.locator("#try-out .pre").textContent({ timeout: 1000 }).catch(() => "")), "(at 0.8 s:", JSON.stringify(preview) + ")");
  await page.mouse.up();
  await page.waitForFunction(() => !document.querySelector("#try-out .pre") && document.getElementById("try-out").textContent.trim(), null, { timeout: 60_000 });
  const text = (await page.locator("#try-out").textContent()).trim();
  console.log("final text:", JSON.stringify(text));
  console.log("status:", await page.locator("#try-status").textContent());
  const missing = words.filter((w) => !text.toLowerCase().includes(w.toLowerCase()));
  ok = missing.length === 0 && !fellBack;
  if (!ok) console.log("missing words:", missing.join(", "));
} catch (e) {
  console.log("error:", e.message);
} finally {
  console.log(logs.join("\n"));
  await browser.close();
}
process.exit(ok ? 0 : 1);
