# Choosing a model

```bash
voiceio models list           # known models, your fine-tunes, and which is active
voiceio models bench          # score candidates on your held-out clips (after `voiceio learn build`)
voiceio models use medium     # switch
```

| Model | Runtime | License | Notes |
|---|---|---|---|
| `small` | faster-whisper | MIT | Default. About 5× realtime on a laptop CPU |
| `medium` | faster-whisper | MIT | Better on names, about 2.5× slower |
| `large-v3-turbo` | faster-whisper | MIT | Best Whisper for dictation; wants a GPU. Also the default `learn` teacher |
| `distil-large-v3` | faster-whisper | MIT | English only |
| `sherpa:parakeet-tdt-0.6b-v3` | sherpa-onnx | CC-BY-4.0 | Experimental. Fast, but no vocabulary biasing yet |
| `whispercpp:http://…` | whisper.cpp server | depends on the model | Experimental. E.g. a Vulkan build on an AMD iGPU |

Pick on your own audio, not leaderboards: for dictation full of names, how well a model can be *biased* matters more than its benchmark WER.

**Licenses.** voiceio ships no model weights. Each model is downloaded from its own source on first use, under its own license. The defaults are permissive (MIT). A fine-tune you make with `voiceio learn` inherits the license of the base model it was trained from.
