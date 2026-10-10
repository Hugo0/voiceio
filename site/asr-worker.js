// In-browser speech recognition for the landing page's "try it" demo. The model
// is downloaded once (browser cache) and runs in this worker on WASM, threaded
// when the page is cross-origin isolated (see _headers). Audio never leaves
// the tab.
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
  postMessage({ type: "ready", model: "moonshine-base", threads: env.backends.onnx.wasm.numThreads || 1 });
}

self.onmessage = async ({ data }) => {
  if (data.type === "load") {
    try { await load(); } catch (err) { report("load", err); postMessage({ type: "error", message: String((err && err.message) || err) }); }
  } else if (data.type === "decode") {
    const t0 = performance.now();
    try {
      // Moonshine can loop on silence or noise; cap output by audio length.
      const { text } = await asr(data.audio, { max_new_tokens: Math.ceil((data.audio.length / 16000) * 7) + 8 });
      postMessage({ type: "text", id: data.id, final: data.final, text: text.trim(), ms: performance.now() - t0 });
    } catch (err) {
      report("decode", err);
      postMessage({ type: "text", id: data.id, final: data.final, text: "", ms: performance.now() - t0 });
    }
  }
};
