"""IBus input method engine for VoiceIO."""
from __future__ import annotations

import os
from pathlib import Path

SOCKET_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "voiceio-ibus.sock"
READY_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "voiceio-ibus.ready"
