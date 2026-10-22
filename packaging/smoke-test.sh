#!/usr/bin/env bash
# Checks an installed voiceio package (.deb or AUR) on a clean system: the
# CLI runs, every heavy native library imports, Whisper decodes, and the
# distro python3 can load the IBus engine from the bundle. Run as root in a
# fresh container after installing the package (see package-test.yml).
set -euo pipefail

export HOME=$(mktemp -d)   # a first-run user, nothing configured
bundle=/usr/lib/voiceio
site=$(ls -d "$bundle"/lib/python3.*/site-packages)

echo "== voiceio --version"
voiceio --version
command -v voiceio-toggle

echo "== bundled libraries"
"$bundle/bin/python" - <<'PY'
import ctranslate2, faster_whisper, onnxruntime, sounddevice, evdev, av, numpy
print("ctranslate2", ctranslate2.__version__, "| faster-whisper", faster_whisper.__version__,
      "| onnxruntime", onnxruntime.__version__, "| av", av.__version__)
PY

echo "== whisper decode (tiny.en, one second of silence)"
"$bundle/bin/python" - <<'PY'
import numpy as np
from faster_whisper import WhisperModel
m = WhisperModel("tiny.en", device="cpu", compute_type="int8")
segs, info = m.transcribe(np.zeros(16000, dtype=np.float32))
print("decoded, duration", info.duration, "segments", len(list(segs)))
PY

echo "== distro python3 loads the IBus engine from the bundle"
PYTHONPATH="$site" python3 -c "
import gi; gi.require_version('IBus', '1.0'); from gi.repository import IBus
import voiceio.ibus.engine; print('ibus engine module ok')"

echo "== voiceio doctor (no audio or display here: failures there are expected)"
voiceio doctor 2>&1 | tail -n 40 || true
voiceio doctor --report 2>&1 | tail -n 12 || true   # no tty: prints, never posts

echo "smoke test passed"
