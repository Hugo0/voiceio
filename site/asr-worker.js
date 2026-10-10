// In-browser speech recognition for the landing page's "try it" demo. The model
// is downloaded once (browser cache) and runs in this worker; audio never
// leaves the tab.
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";

const CHOICES = {
  webgpu: { model: "onnx-community/moonshine-base-ONNX", dtype: { encoder_model: "fp32", decoder_model_merged: "q4" } },
  wasm: { model: "onnx-community/moonshine-base-ONNX", dtype: "q8" },
};
let asr = null;

async function load(device) {
  const { model, dtype } = CHOICES[device];
  asr = await pipeline("automatic-speech-recognition", model, {
    device, dtype,
    progress_callback: (p) => p.status === "progress" && postMessage({ type: "progress", file: p.file, loaded: p.loaded, total: p.total }),
  });
  await asr(new Float32Array(16000));  // warm-up: compiles kernels before the first real decode
  postMessage({ type: "ready", device, model: model.split("/")[1].replace("-ONNX", "") });
}

self.onmessage = async ({ data }) => {
  if (data.type === "load") {
    try {
      await load(data.webgpu ? "webgpu" : "wasm");
    } catch (err) {
      if (!data.webgpu) return postMessage({ type: "error", message: String(err) });
      try { await load("wasm"); } catch (err2) { postMessage({ type: "error", message: String(err2) }); }
    }
  } else if (data.type === "decode") {
    const t0 = performance.now();
    // Moonshine can loop on silence or noise; cap output by audio length.
    const { text } = await asr(data.audio, { max_new_tokens: Math.ceil((data.audio.length / 16000) * 7) + 8 });
    postMessage({ type: "text", id: data.id, final: data.final, text: text.trim(), ms: performance.now() - t0 });
  }
};
