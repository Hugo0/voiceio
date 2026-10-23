# Choosing a model

```bash
voiceio models list                         # known models, your fine-tunes, the active one and the default
voiceio models bench                        # score candidates on your held-out clips (after `voiceio learn build`)
voiceio models use parakeet-tdt-0.6b-v2     # switch (downloads it); `use auto` goes back to the default
voiceio models sherpa                       # install sherpa-onnx with the Parakeet fixes (see below)
```

| Model | Runtime | License | Notes |
|---|---|---|---|
| `parakeet-tdt-0.6b-v2` | sherpa-onnx | CC-BY-4.0 | **Default** where it can be biased (see below). English only. About 2× faster than `small`, every proper noun of your vocabulary as a hotword |
| `small` | faster-whisper | MIT | Default otherwise. About 5× realtime on a laptop CPU; ~35 vocabulary terms at most. What `voiceio learn` fine-tunes |
| `medium` | faster-whisper | MIT | Better on names, about 2.5× slower |
| `large-v3-turbo` | faster-whisper | MIT | Best Whisper for dictation; wants a GPU. Also the default `learn` teacher |
| `distil-large-v3` | faster-whisper | MIT | English only |
| `whispercpp:http://…` | whisper.cpp server | depends on the model | Experimental. E.g. a Vulkan build on an AMD iGPU |

Pick on your own audio, not leaderboards: for dictation full of names, how well a model can be *biased* matters more than its benchmark WER.

## Parakeet and sherpa-onnx

Parakeet's vocabulary biasing (hotwords, with no size limit) needs
`modified_beam_search`, which drops speech on NeMo TDT models in sherpa-onnx
releases up to 1.13.8. voiceio checks the installed sherpa-onnx: with the
fix (`length_reward`), `[model] name` left unset or `"auto"` means Parakeet;
without it, the default stays `small`, and an explicitly chosen Parakeet
decodes greedily with no biasing (`voiceio doctor` says so). When an upstream
release ships the fix, the default flips by itself.

Until then, `voiceio models sherpa` installs a build with the fix and with
pre-tokenized hotwords (sherpa's own hotword tokenization splits most proper
nouns differently from the model, so they would never match). It is built
from `Hugo0/sherpa-onnx@voiceio-patched` for Linux x86_64, Python 3.10–3.13,
and verified against its SHA256SUMS before installing.

Only proper nouns (terms with a capital letter) become Parakeet hotwords:
biasing toward lowercase jargon made the model insert those words where they
weren't said. Parakeet writes cased, punctuated text and spells numbers out;
voiceio's number conversion and cleanup handle that as for Whisper. Audio is
decoded in pieces of at most 30 s, cut at quiet spots.

The model comes from the sherpa-onnx GitHub release (checked against a
pinned sha256) into `~/.cache/voiceio/models/`, never from Hugging Face.

**Fine-tunes.** `voiceio learn` trains Whisper. While Parakeet is the decoder,
`voiceio learn maintain` keeps labeling your clips but doesn't retrain (it
says so); `voiceio learn train` still trains on request, and `voiceio learn
promote RUN` switches you back to a fine-tune. An explicit `[model] name`,
such as a promoted fine-tune's path, always wins over the default.

**Licenses.** voiceio ships no model weights. Each model is downloaded from its own source on first use, under its own license. The Whisper models are MIT; Parakeet is CC-BY-4.0 (NVIDIA). A fine-tune you make with `voiceio learn` inherits the license of the base model it was trained from.
