// In-browser speech recognition for the landing page's "try it" demo. The model
// is downloaded once (browser cache) and runs in this worker; audio never
// leaves the tab.
//
// WASM only, threaded when the page is cross-origin isolated (see _headers).
// WebGPU was measured and lost for this model (moonshine-base, 2026-10-10, AMD
// RDNA3, Chromium/Vulkan): the fp16 encoder produced garbage, and fp32/q4
// encoders were correct but slower than threaded WASM (final 2.2 s vs 1.3 s,
// partials 0.8-2.8 s vs 0.1-0.5 s). The model is too small to win back the
// GPU round-trips. Revisit for a larger model.
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";

const MODEL = "onnx-community/moonshine-base-ONNX";
const report = (where, err) => postMessage({ type: "log", message: where + ": " + ((err && (err.stack || err.message)) || err) });
self.addEventListener("error", (e) => report("worker error", e.message));
self.addEventListener("unhandledrejection", (e) => report("unhandled rejection", e.reason));

let asr = null;

async function load() {
  if (self.crossOriginIsolated) env.backends.onnx.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
  asr = await pipeline("automatic-speech-recognition", MODEL, {
    device: "wasm", dtype: "q8",
    progress_callback: (p) => p.status === "progress" && postMessage({ type: "progress", file: p.file, loaded: p.loaded, total: p.total }),
  });
  await asr(new Float32Array(16000));  // warm-up before the first real decode
  postMessage({ type: "ready", model: "moonshine-base" });
}

self.onmessage = async ({ data }) => {
  if (data.type === "load") {
    try { await load(); } catch (err) { report("load", err); postMessage({ type: "error", message: String((err && err.message) || err) }); }
  } else if (data.type === "decode") {
    const t0 = performance.now();
    let text = "";
    try {
      // Moonshine can loop on silence or noise; cap output by audio length.
      text = (await asr(data.audio, { max_new_tokens: Math.ceil((data.audio.length / 16000) * 7) + 8 })).text.trim();
    } catch (err) {
      report("decode", err);
    }
    postMessage({ type: "text", id: data.id, final: data.final, text, ms: performance.now() - t0 });
  }
};
