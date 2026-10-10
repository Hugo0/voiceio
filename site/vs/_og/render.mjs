// Renders site/og/<slug>.png (1200x630) from template.html with headless Chromium.
// Usage: PW_DIR=<dir with node_modules/playwright-core> CHROME=<chrome binary> node render.mjs [slug...]
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";
import { mkdirSync, statSync } from "node:fs";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(join(process.env.PW_DIR || here, "x.js"));
const { chromium } = require("playwright-core");

export const CARDS = {
  "vs":                        { k: "// compare", t: "voiceio vs other dictation tools", s: "Local, open-source dictation for Linux, side by side with the alternatives." },
  "voxtype":                   { k: "// compare", t: "voiceio vs Voxtype", s: "Two local, open-source Linux dictation tools. Where each one fits." },
  "aqua-voice":                { k: "// compare", t: "voiceio vs Aqua Voice", s: "Cloud dictation app vs local, open-source Linux dictation." },
  "wispr-flow":                { k: "// compare", t: "voiceio vs Wispr Flow", s: "Cloud dictation app vs local, open-source Linux dictation." },
  "superwhisper":              { k: "// compare", t: "voiceio vs Superwhisper", s: "Whisper-based dictation for Apple platforms and Windows vs Linux." },
  "dragon":                    { k: "// compare", t: "voiceio vs Dragon", s: "Commercial speech recognition for Windows vs local, open-source Linux dictation." },
  "nerd-dictation":            { k: "// compare", t: "voiceio vs nerd-dictation", s: "Two offline, open-source Linux dictation tools. Where each one fits." },
  "talon":                     { k: "// compare", t: "voiceio vs Talon", s: "Hands-free computer control vs dictation that learns your voice." },
  "whisper":                   { k: "// compare", t: "voiceio and Whisper", s: "voiceio is built on Whisper. What it adds on top." },
  "linux-voice-dictation":     { k: "// guide", t: "Voice dictation on Linux", s: "Speak into any app on Wayland or X11. Local, private, open source." },
  "offline-speech-to-text":    { k: "// guide", t: "Offline speech to text", s: "Dictation that runs on your own computer, with no cloud and no account." },
  "whisper-dictation-wayland": { k: "// guide", t: "Whisper dictation on Wayland", s: "Live Whisper dictation into any Wayland app through IBus." },
};

const out = join(here, "..", "..", "og");
mkdirSync(out, { recursive: true });
const only = process.argv.slice(2);
const browser = await chromium.launch({ executablePath: process.env.CHROME });
const page = await browser.newPage({ viewport: { width: 1200, height: 630 } });
for (const [slug, c] of Object.entries(CARDS)) {
  if (only.length && !only.includes(slug)) continue;
  const url = pathToFileURL(join(here, "template.html")) + "?" + new URLSearchParams(c);
  await page.goto(url);
  await page.waitForSelector("body[data-ready]");
  const file = join(out, slug + ".png");
  await page.screenshot({ path: file, type: "png" });
  console.log(slug, Math.round(statSync(file).size / 1024) + " KB");
}
await browser.close();
