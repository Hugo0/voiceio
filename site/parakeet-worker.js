// In-browser speech recognition for the landing page's "try it" demo:
// NVIDIA Parakeet-TDT 110M (int8) through sherpa-onnx compiled to WebAssembly.
// Same message protocol as asr-worker.js (the Moonshine fallback): in "load",
// out "progress" / "ready" / "error" / "log"; in "decode", out "text".
//
// Everything is same-origin: the sherpa-onnx build in sherpa/ and the model in
// models/ (the encoder split into parts under the 25 MiB static-asset cap; see
// scripts/site/fetch-demo-model.sh). Files are kept in Cache Storage after the
// first visit. The sherpa-onnx wasm build is single-threaded, so this worker
// does not need cross-origin isolation.
const BASE = new URL("./", self.location.href).href;
const CACHE = "voiceio-parakeet-v1";

const report = (where, err) => postMessage({ type: "log", message: where + ": " + ((err && (err.stack || err.message)) || err) });
self.addEventListener("error", (e) => report("worker error", e.message));
self.addEventListener("unhandledrejection", (e) => report("unhandled rejection", e.reason));

let recognizer = null;
let Module = null;

async function cached(url) {
  try {
    const cache = await caches.open(CACHE);
    const hit = await cache.match(url);
    if (hit) return { res: hit, cache: null };
    return { res: await fetch(url), cache };
  } catch (_) {  // no Cache Storage (private window, blocked storage): plain fetch
    return { res: await fetch(url), cache: null };
  }
}

// Fetch one file into `out` at `offset`, reporting progress as `name`.
async function fetchInto(url, name, size, out, offset) {
  const { res, cache } = await cached(url);
  if (!res.ok) throw new Error(url.slice(BASE.length) + ": HTTP " + res.status);
  if (cache) cache.put(url, res.clone()).catch((e) => report("cache " + name, e));
  const reader = res.body.getReader();
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (got + value.length > size) throw new Error(name + ": larger than expected");
    out.set(value, offset + got);
    got += value.length;
    postMessage({ type: "progress", file: name, loaded: got, total: size });
  }
  if (got !== size) throw new Error(name + ": got " + got + " of " + size + " bytes");
}

async function fetchFile(dir, f) {
  const buf = new Uint8Array(f.size);
  let offset = 0;
  for (const part of f.parts) {
    await fetchInto(dir + part.name, part.name, part.size, buf, offset);
    offset += part.size;
  }
  return buf;
}

// `opts` exists for the benchmark (other builds, models, beam search).
async function load(opts) {
  if (typeof WebAssembly !== "object") throw new Error("no WebAssembly");
  const SHERPA = BASE + (opts.sherpa || "sherpa/");
  const MODEL = BASE + (opts.model || "models/parakeet-tdt-110m-int8/");
  const json = async (url) => (await fetch(url, { cache: "no-cache" })).json();
  const [runtime, manifest] = await Promise.all([json(SHERPA + "manifest.json"), json(MODEL + "manifest.json")]);
  // Announce every file up front so the page's progress bar counts the total.
  for (const f of [runtime.wasm, ...manifest.files]) for (const p of f.parts) postMessage({ type: "progress", file: p.name, loaded: 0, total: p.size });

  const wasmBinary = fetchFile(SHERPA, runtime.wasm);
  const files = Promise.all(manifest.files.map(async (f) => [f.name, await fetchFile(MODEL, f)]));
  importScripts(SHERPA + "sherpa-onnx-wasm-web.js", SHERPA + "sherpa-onnx-asr.js");
  Module = await self.SherpaOnnx({ wasmBinary: (await wasmBinary).buffer, print: () => {}, printErr: (t) => report("sherpa", t) });

  Module.FS.mkdir("/m");
  for (const [name, data] of await files) Module.FS.writeFile("/m/" + name, data, { canOwn: true });
  recognizer = new OfflineRecognizer({
    featConfig: { sampleRate: 16000, featureDim: 80 },
    modelConfig: {
      transducer: { encoder: "/m/encoder.onnx", decoder: "/m/decoder.onnx", joiner: "/m/joiner.onnx" },
      tokens: "/m/tokens.txt",
      numThreads: 1,
      provider: "cpu",
      debug: 0,
      modelType: "nemo_transducer",
    },
    decodingMethod: opts.decodingMethod || "greedy_search",
  }, Module);
  if (!recognizer.handle) throw new Error("sherpa-onnx could not create the recognizer");
  // The session holds its own copy of the weights; drop the file copies.
  for (const f of manifest.files) Module.FS.unlink("/m/" + f.name);
  decode(new Float32Array(16000));  // warm-up before the first real decode
  postMessage({ type: "ready", model: "parakeet-tdt-110m" });
}

function decode(audio) {
  const stream = recognizer.createStream();
  try {
    stream.acceptWaveform(16000, audio);
    recognizer.decode(stream);
    return recognizer.getResult(stream).text.trim();
  } finally {
    stream.free();
  }
}

self.onmessage = async ({ data }) => {
  if (data.type === "load") {
    try { await load(data.opts || {}); } catch (err) { report("load", err); postMessage({ type: "error", message: String((err && err.message) || err) }); }
  } else if (data.type === "decode") {
    const t0 = performance.now();
    let text = "";
    try { text = decode(data.audio); } catch (err) { report("decode", err); }
    postMessage({ type: "text", id: data.id, final: data.final, text, ms: performance.now() - t0 });
  }
};
