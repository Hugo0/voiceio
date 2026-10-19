"""Fine-tune Whisper on the user's own speech (LoRA), export for faster-whisper.

    export   dataset → a self-contained folder of ≤28s WAV windows + JSONL
             (copy it to a GPU box; nothing else from the machine is needed)
    train    LoRA on the attention/MLP projections of an HF Whisper checkpoint
    convert  merge the adapter, convert to CTranslate2 int8 — the format the
             live decoder loads, so hotwords and everything else keep working

Heavy dependencies (torch, transformers, peft) are imported lazily and live
in the `[train]` extra. LoRA rather than a full fine-tune: it forgets far less
of the base model's general ability (arXiv 2408.10680) and a run on a few
hours of audio takes under an hour on one consumer GPU.

On CPU the default is decoder-only: the encoder is frozen, so its output for
each window is computed once and cached, and every step trains only the
decoder — where vocabulary spelling is learned. Measured on a Ryzen 7940HS
under load (whisper-small): ~1.1s per window vs ~18s for a full step. Full LoRA also adapts
the encoder to a voice and microphone; use it where there is a GPU.

Training is resumable: the loop checkpoints its progress about once a minute
(and the frozen encoder's outputs live in a file), so a run that is stopped
— the scheduled one is, the moment you come back to the computer — carries
on where it was instead of starting over.

A fine-tuned model inherits the base model's license; Whisper is MIT.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voiceio.learn.store import read_split
from voiceio.learn.teacher import load_wav

log = logging.getLogger(__name__)

# HF checkpoints matching faster-whisper's model names.
HF_BASE = {
    "tiny": "openai/whisper-tiny", "base": "openai/whisper-base",
    "small": "openai/whisper-small", "medium": "openai/whisper-medium",
    "large-v3": "openai/whisper-large-v3", "large-v3-turbo": "openai/whisper-large-v3-turbo",
    "tiny.en": "openai/whisper-tiny.en", "base.en": "openai/whisper-base.en",
    "small.en": "openai/whisper-small.en", "medium.en": "openai/whisper-medium.en",
}


@dataclass
class TrainConfig:
    base: str = "small"
    epochs: int = 4
    lr: float = 1e-4            # LoRA tolerates a much higher rate than full FT
    batch_size: int = 8
    lora_r: int = 32
    lora_alpha: int = 64
    language: str = "en"
    seed: int = 0
    decoder_only: bool | None = None    # None: decoder-only on CPU, full on GPU
    # Share of windows trained in Whisper's timestamp mode (as Distil-Whisper
    # does): dictation decodes with timestamps, so both modes must adapt.
    timestamp_share: float = 0.5
    # Share of windows trained behind the user's hotword prompt, exactly as
    # faster-whisper prepends it to every live decode — so fine-tuning keeps
    # (and learns with) the biasing channel instead of eroding it.
    hotwords: str = ""
    prompt_share: float = 0.5
    threads: int = 0            # CPU threads; 0 = half the logical CPUs


# Full: every attention/MLP projection. Decoder-only additionally skips the
# cross-attention key/value projections — they re-project all 1500 encoder
# frames each step, so adapting them doubled step time (2.15s → 1.13s per
# window, Ryzen 7940HS). They are how the decoder reads acoustics; vocabulary
# lives in self-attention, cross-attention query/output and the MLPs.
_FULL_TARGETS = r".*\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)"
_DECODER_TARGETS = (r".*decoder\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)"
                    r"|encoder_attn\.(q_proj|out_proj)|fc1|fc2)")


TRAIN_HINT = "pip install 'python-voiceio[train]'"


def missing_train_deps() -> list[str]:
    """Training packages that aren't installed (the `train` extra)."""
    import importlib.util
    return [m for m in ("torch", "transformers", "peft") if importlib.util.find_spec(m) is None]


def require_train_deps() -> None:
    missing = missing_train_deps()
    if missing:
        raise RuntimeError(
            f"training needs {', '.join(missing)}: pip install 'python-voiceio[train]'")


# ── export ───────────────────────────────────────────────────────────────

def export(dataset: Path, out: Path, *, extra_train_rows: list[dict] = (),
           hotwords: str = "") -> dict:
    """Cut each train/dev window to its own 16 kHz WAV under `out`, with
    `{split}.jsonl` rows {"audio": relpath, "text"[, "parts"]} and the live
    hotword prompt in hotwords.txt. Returns counts."""
    out.mkdir(parents=True, exist_ok=True)
    (out / HOTWORDS_FILE).write_text(hotwords, encoding="utf-8")
    counts = {}
    cache: dict[str, np.ndarray] = {}
    for split in ("train", "dev"):
        rows = read_split(dataset, split) + (list(extra_train_rows) if split == "train" else [])
        (out / split).mkdir(exist_ok=True)
        lines = []
        for i, row in enumerate(rows):
            src = row["file_name"]
            if src not in cache:
                cache.clear()  # rows are grouped by clip; keep one decoded at a time
                cache[src] = load_wav(Path(src))
            audio = cache[src][int(row["start"] * 16000):int(row["end"] * 16000)]
            if len(audio) < 1600:
                continue
            rel = f"{split}/{i:06d}.wav"
            with wave.open(str(out / rel), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
            out_row = {"audio": rel, "text": row["text"]}
            if row.get("parts"):
                out_row["parts"] = row["parts"]
            lines.append(json.dumps(out_row, ensure_ascii=False))
        (out / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts[split] = len(lines)
    return counts


# ── targets ──────────────────────────────────────────────────────────────

HOTWORDS_FILE = "hotwords.txt"
_MAX_DECODER_TOKENS = 448   # Whisper's decoder positions; HF raises past it
_TS_STEP = 0.02     # Whisper timestamp resolution
_TS_MAX = 1500      # <|30.00|>


def target_ids(tokenizer, row: dict, *, timestamps: bool) -> list[int]:
    """Decoder target for one window, as Whisper itself would emit it.

    Text gets a leading space: Whisper's output always starts with one, and
    "hello" without it tokenizes differently ("he"+"llo"), so the model would
    be trained on sequences it never produces. With `timestamps` (and segment
    `parts` available) the target interleaves <|t|> tokens around each part —
    the format dictation decodes in — instead of <|notimestamps|> + text.
    """
    text = " " + row["text"].strip()
    ids = tokenizer(text).input_ids     # [SOT, lang, task, <|notimestamps|>, …, EOT]
    parts = row.get("parts")
    if not timestamps or not parts:
        return ids
    no_ts = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    out = ids[:ids.index(no_ts)]

    def ts(t: float) -> int:
        return no_ts + 1 + min(_TS_MAX, max(0, round(t / _TS_STEP)))

    for start, end, part in parts:
        out += [ts(start), *tokenizer(" " + part.strip(), add_special_tokens=False).input_ids,
                ts(end)]
    return out + [tokenizer.eos_token_id]


def _row_draws(key: str) -> tuple[float, float]:
    """Two stable uniform draws per row (dev targets must not vary by epoch)."""
    import hashlib
    h = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32, int.from_bytes(h[4:8], "big") / 2**32


def prompt_ids(tokenizer, hotwords: str, max_length: int = 448) -> list[int]:
    """faster-whisper's hotword prefix: <|startofprev|> + " hotwords" (capped)."""
    tokens = tokenizer(" " + hotwords.strip(), add_special_tokens=False).input_ids
    if len(tokens) >= max_length // 2:
        tokens = tokens[:max_length // 2 - 1]
    return [tokenizer.convert_tokens_to_ids("<|startofprev|>"), *tokens]


def decoder_batch(seqs: list[tuple[list[int], int]], pad: int):
    """(sequence, n_prompt) pairs → (decoder_input_ids, labels) lists, right-
    padded. A sequence is [prompt…] SOT … EOT; the model reads all but the
    last token and is scored on predicting each next one — except the prompt
    and the SOT that follows it, which are context, not output."""
    inputs, labels = [], []
    for seq, n_prompt in seqs:
        inputs.append(seq[:-1])
        labels.append([-100] * n_prompt + seq[n_prompt + 1:])
    width = max(len(x) for x in inputs)
    return ([x + [pad] * (width - len(x)) for x in inputs],
            [y + [-100] * (width - len(y)) for y in labels])


# ── setup ────────────────────────────────────────────────────────────────

def _prefix(vocab_size: int, cfg: TrainConfig) -> dict:
    """Tokenizer prefix matching what faster-whisper feeds at inference. An
    English-only model (vocabulary < 51865, CTranslate2's own test) is
    decoded from <|startoftranscript|> alone: training it behind
    <|en|><|transcribe|> would teach a prefix it never sees."""
    return {"language": cfg.language, "task": "transcribe"} if vocab_size >= 51865 else {}


def lr_schedule(total_steps: int, warmup: float = 0.1):
    """Linear warmup over the first `warmup` share of steps, then linear decay
    to zero — a few hundred steps on a few hours of audio overshoot without."""
    warm = max(1, int(total_steps * warmup))

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))
    return factor


# ── resumable state ──────────────────────────────────────────────────────

CHECKPOINT = "checkpoint.pt"
ENCODER_CACHE = "encoder-cache"     # .f16 (outputs), .filled (flags), .json (what for)
CHECKPOINT_SECS = 60.0


def fingerprint(folder: Path, cfg: TrainConfig, **resolved) -> str:
    """What saved progress is valid for: these windows, targets and settings.
    Anything else starts over rather than resuming into a mismatch."""
    import dataclasses
    import hashlib
    h = hashlib.sha256()
    for name in ("train.jsonl", "dev.jsonl", HOTWORDS_FILE):
        path = folder / name
        h.update(path.read_bytes() if path.exists() else b"")
        h.update(b"\0")
    settings = {k: v for k, v in dataclasses.asdict(cfg).items() if k != "threads"}
    h.update(json.dumps({**settings, **resolved}, sort_keys=True).encode())
    return h.hexdigest()[:16]


class EncoderCache:
    """Frozen-encoder output per window, in an fp16 file mapped into memory.

    Computed once per run and kept across interruptions: it is a large share
    of a CPU epoch. Mapped rather than held: at 1500 frames × d_model per
    window it is gigabytes (~4 GB for 1700 windows of whisper-small), and
    file-backed pages are dropped under memory pressure instead of pushing
    the desktop into swap."""

    def __init__(self, base: Path, keys: list[str], *, frames: int, width: int,
                 fingerprint: str):
        self.index = {k: i for i, k in enumerate(keys)}
        shape = (max(1, len(keys)), frames, width)
        meta = {"fingerprint": fingerprint, "shape": list(shape)}
        data, filled, info = self.paths(base)
        try:
            reuse = (json.loads(info.read_text()) == meta
                     and data.stat().st_size == int(np.prod(shape)) * 2
                     and filled.stat().st_size == shape[0])
        except (OSError, ValueError):
            reuse = False
        if not reuse:
            info.unlink(missing_ok=True)
        mode = "r+" if reuse else "w+"
        self.data = np.memmap(data, dtype=np.float16, mode=mode, shape=shape)
        self.filled = np.memmap(filled, dtype=np.uint8, mode=mode, shape=(shape[0],))
        if not reuse:
            info.write_text(json.dumps(meta))
        self.reused = int(self.filled.sum()) if reuse else 0

    @staticmethod
    def paths(base: Path) -> tuple[Path, Path, Path]:
        return tuple(base.with_name(base.name + ext) for ext in (".f16", ".filled", ".json"))

    def missing(self, keys: list[str]) -> list[str]:
        return [k for k in keys if not self.filled[self.index[k]]]

    def put(self, keys: list[str], values: np.ndarray) -> None:
        idx = [self.index[k] for k in keys]
        self.data[idx] = values
        self.data.flush()           # the outputs reach the file before their flags
        self.filled[idx] = 1
        self.filled.flush()

    def get(self, keys: list[str]) -> np.ndarray:
        return self.data[[self.index[k] for k in keys]]


def discard_resume_state(out: Path) -> None:
    """Drop a finished run's checkpoint and encoder cache (gigabytes)."""
    for path in (out / CHECKPOINT, *EncoderCache.paths(out / ENCODER_CACHE)):
        path.unlink(missing_ok=True)


def _save_checkpoint(path: Path, state: dict) -> None:
    import torch
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)       # stopped mid-save: the previous one stands


def _load_checkpoint(path: Path, fp: str) -> dict | None:
    import torch
    if not path.exists():
        return None
    try:
        state = torch.load(path, weights_only=True)
    except Exception as e:  # noqa: BLE001 — unreadable: start over, never fail the run
        log.warning("ignoring unreadable checkpoint %s: %s", path, e)
        return None
    if state.get("fingerprint") != fp:
        print("  saved progress is for other data or settings: starting over", flush=True)
        return None
    return state


def fit(model, forward, train_rows: list[dict], dev_rows: list[dict], cfg: TrainConfig, *,
        checkpoint: Path, fp: str, order_rng: np.random.Generator,
        mix_rng: np.random.Generator, save_best, checkpoint_secs: float = CHECKPOINT_SECS,
        clock=time.monotonic) -> float:
    """The training loop; returns the best dev loss. `forward(chunk, fixed=)`
    gives a batch's loss; `save_best()` keeps the adapter of the best epoch.

    Resumable at any point: every `checkpoint_secs` and after each epoch it
    saves the trainable weights, optimizer, schedule and every random state
    (the epoch's shuffle is replayed from the state it started with), so a
    resumed run takes exactly the steps an uninterrupted one would have.
    At most `checkpoint_secs` of work is lost to a stop."""
    import torch
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in trainable]
    steps = -(-len(train_rows) // cfg.batch_size)
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule(steps * cfg.epochs))
    best, first_epoch, done = float("inf"), 0, 0
    epoch_rng = order_rng.bit_generator.state
    if state := _load_checkpoint(checkpoint, fp):
        with torch.no_grad():
            for n, p in trainable:
                p.copy_(state["params"][n])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        mix_rng.bit_generator.state = state["mix_rng"]
        torch.set_rng_state(state["torch_rng"])
        epoch_rng, best = state["order_rng"], state["best"]
        first_epoch, done = state["epoch"], state["step"]
        print(f"  resuming at epoch {first_epoch + 1}/{cfg.epochs} step {done}/{steps}"
              if first_epoch < cfg.epochs else "  resuming: training had finished", flush=True)

    def save(epoch: int, step: int) -> None:
        _save_checkpoint(checkpoint, {
            "fingerprint": fp, "epoch": epoch, "step": step, "best": best,
            "params": {n: p.detach().clone() for n, p in trainable},
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "order_rng": epoch_rng, "mix_rng": mix_rng.bit_generator.state,
            "torch_rng": torch.get_rng_state()})

    last_save = clock()
    for epoch in range(first_epoch, cfg.epochs):
        order_rng.bit_generator.state = epoch_rng
        model.train()
        order = [train_rows[i] for i in order_rng.permutation(len(train_rows))]
        skip = done if epoch == first_epoch else 0
        t0, timed = clock(), 0
        for step, i in enumerate(range(0, len(order), cfg.batch_size), 1):
            if step <= skip:
                continue
            loss = forward(order[i:i + cfg.batch_size])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            timed += 1
            if step % 10 == 0 or step == steps:
                eta = (clock() - t0) / timed * (steps - step)
                print(f"  epoch {epoch + 1}/{cfg.epochs} step {step}/{steps} "
                      f"loss {loss.item():.3f}  ~{eta / 60:.0f} min left in epoch", flush=True)
            if step < steps and clock() - last_save >= checkpoint_secs:
                save(epoch, step)
                last_save = clock()
        model.eval()
        with torch.no_grad():
            losses = [forward(dev_rows[i:i + cfg.batch_size], fixed=True).item()
                      for i in range(0, len(dev_rows), cfg.batch_size)]
        # No dev split (tiny dataset): keep the latest epoch.
        dev = float(np.mean(losses)) if losses else float("-inf")
        log.info("epoch %d dev loss %.4f", epoch + 1, dev)
        print(f"  epoch {epoch + 1}/{cfg.epochs}: dev loss {dev:.4f}", flush=True)
        if dev <= best or not losses:
            best = dev
            save_best()
        epoch_rng = order_rng.bit_generator.state     # where the next shuffle starts
        save(epoch + 1, 0)
        last_save = clock()
    return best


# ── fine-tune ────────────────────────────────────────────────────────────

def train(folder: Path, out: Path, cfg: TrainConfig = TrainConfig()) -> Path:
    """LoRA-fine-tune on an exported folder; returns the merged HF model dir.
    Resumes from `out`'s checkpoint when one matches this folder and config."""
    require_train_deps()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(cfg.threads or max(1, (os.cpu_count() or 2) // 2))
    decoder_only = cfg.decoder_only if cfg.decoder_only is not None else device == "cpu"
    base_id = HF_BASE.get(cfg.base, cfg.base)
    model = WhisperForConditionalGeneration.from_pretrained(base_id)
    processor = WhisperProcessor.from_pretrained(base_id, **_prefix(model.config.vocab_size, cfg))
    model.config.forced_decoder_ids = None
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.05,
        target_modules=_DECODER_TARGETS if decoder_only else _FULL_TARGETS,
    )).to(device)
    model.print_trainable_parameters()
    encoder = model.get_base_model().model.encoder

    def load_rows(split: str) -> list[dict]:
        path = folder / f"{split}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    train_rows, dev_rows = load_rows("train"), load_rows("dev")
    out.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(folder, cfg, base_id=base_id, decoder_only=decoder_only, device=device)
    mix_rng = np.random.default_rng(cfg.seed + 1)
    prompt = prompt_ids(processor.tokenizer, cfg.hotwords) if cfg.hotwords.strip() else []

    def targets_for(chunk: list[dict], *, fixed: bool = False):
        """Training rows draw their timestamp/prompt mix at random each epoch;
        dev rows (`fixed`) get the same mix every epoch, so epochs compare."""
        seqs = []
        for r in chunk:
            u, v = _row_draws(r["audio"]) if fixed else (mix_rng.random(), mix_rng.random())
            ids = target_ids(processor.tokenizer, r, timestamps=u < cfg.timestamp_share)
            p = prompt if prompt and v < cfg.prompt_share else []
            if len(p) + len(ids) > _MAX_DECODER_TOKENS:
                p = []                                   # context first
                ids = ids[:_MAX_DECODER_TOKENS - 1] + ids[-1:]   # keep EOT
            seqs.append((p + ids, len(p)))
        dec_in, labels = decoder_batch(seqs, processor.tokenizer.eos_token_id)
        return (torch.tensor(dec_in, device=device), torch.tensor(labels, device=device))

    def features(chunk: list[dict]):
        audio = [load_wav(folder / r["audio"]) for r in chunk]
        feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        return feats.input_features.to(device)

    cache = EncoderCache(
        out / ENCODER_CACHE, list(dict.fromkeys(r["audio"] for r in train_rows + dev_rows)),
        frames=encoder.config.max_source_positions, width=encoder.config.d_model,
        fingerprint=fp) if decoder_only else None
    if cache is not None and cache.reused:
        print(f"  {cache.reused} encoder outputs kept from before", flush=True)

    def encoded(chunk: list[dict]):
        """Frozen-encoder output, computed once per window (fp16, on disk)."""
        keys = [r["audio"] for r in chunk]
        if missing := set(cache.missing(keys)):
            rows = [r for r in chunk if r["audio"] in missing]
            with torch.no_grad():
                hidden = encoder(features(rows)).last_hidden_state.half().cpu().numpy()
            cache.put([r["audio"] for r in rows], hidden)
        return torch.from_numpy(cache.get(keys)).to(device).float()

    def forward(chunk: list[dict], *, fixed: bool = False):
        dec_in, labels = targets_for(chunk, fixed=fixed)
        if decoder_only:
            from transformers.modeling_outputs import BaseModelOutput
            return model(encoder_outputs=BaseModelOutput(last_hidden_state=encoded(chunk)),
                         decoder_input_ids=dec_in, labels=labels).loss
        return model(input_features=features(chunk), decoder_input_ids=dec_in,
                     labels=labels).loss

    print(f"  {len(train_rows)} train / {len(dev_rows)} dev windows on {device}, "
          f"{'decoder-only' if decoder_only else 'full'} LoRA", flush=True)
    best = fit(model, forward, train_rows, dev_rows, cfg, checkpoint=out / CHECKPOINT, fp=fp,
               order_rng=np.random.default_rng(cfg.seed), mix_rng=mix_rng,
               save_best=lambda: model.save_pretrained(out / "adapter"))

    from peft import PeftModel
    merged = PeftModel.from_pretrained(
        WhisperForConditionalGeneration.from_pretrained(base_id), out / "adapter",
    ).merge_and_unload()
    hf_dir = out / "hf"
    merged.save_pretrained(hf_dir)
    processor.save_pretrained(hf_dir)
    # transformers 5 folds this into processor_config.json; CTranslate2 and
    # faster-whisper still read it on its own.
    processor.feature_extractor.save_pretrained(hf_dir)
    (out / "train.json").write_text(json.dumps(
        {"base": base_id, "epochs": cfg.epochs, "lr": cfg.lr, "lora_r": cfg.lora_r,
         "decoder_only": decoder_only, "timestamp_share": cfg.timestamp_share,
         "prompt_share": cfg.prompt_share if prompt else 0.0,
         "best_dev_loss": best}, indent=2) + "\n")
    return hf_dir


def convert(hf_dir: Path, out: Path, quantization: str = "int8") -> Path:
    """HF Whisper → CTranslate2, loadable as `[model] name = "<out>"`."""
    from ctranslate2.converters import TransformersConverter
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    TransformersConverter(str(hf_dir), copy_files=["tokenizer.json", "preprocessor_config.json"]
                          ).convert(str(tmp), quantization=quantization, force=True)
    # Into place whole: a run stopped mid-convert must not leave a model.bin
    # that looks finished.
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    return out


def remote_recipe(folder: Path, run: Path, base: str) -> str:
    """Commands to train on a rented GPU and bring the model back."""
    return f"""\
# 1. copy the exported windows (audio + text only) to the GPU box
rsync -a {folder}/ gpu:voiceio-train/
# 2. on the GPU box
pip install 'python-voiceio[train]'
python -m voiceio.learn.train ~/voiceio-train ~/voiceio-run --base {base}
# 3. bring the converted model back, then: voiceio learn promote {run.name}
rsync -a gpu:voiceio-run/ct2/ {run}/ct2/"""


def main(argv: list[str] | None = None) -> None:
    """`python -m voiceio.learn.train FOLDER OUT [--base small]` (GPU box entry)."""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m voiceio.learn.train")
    ap.add_argument("folder", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--base", default="small")
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--full", action="store_true", help="also adapt the encoder (GPU)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    hotwords_file = args.folder / HOTWORDS_FILE
    hotwords = hotwords_file.read_text(encoding="utf-8") if hotwords_file.exists() else ""
    hf = train(args.folder, args.out, TrainConfig(base=args.base, epochs=args.epochs,
                                                  decoder_only=False if args.full else None,
                                                  hotwords=hotwords))
    convert(hf, args.out / "ct2")
    discard_resume_state(args.out)
    shutil.rmtree(args.out / "hf", ignore_errors=True)
    print(f"done: {args.out / 'ct2'}")


if __name__ == "__main__":
    main()
