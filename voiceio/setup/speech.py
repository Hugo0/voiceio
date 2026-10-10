"""Whisper model choice, download, and the live mic/transcription test."""
from __future__ import annotations

import sys
import time

from voiceio.setup.hotkey import _test_hotkey
from voiceio.setup.ui import Spinner, _ask_choice
from voiceio.ui import BOLD, CYAN, DIM, GREEN, RESET, YELLOW


MODELS = [
    ("tiny", "75 MB", "Fastest, basic accuracy"),
    ("base", "150 MB", "Fast, weak on names/technical terms"),
    ("small", "500 MB", "Good accuracy, ~5x realtime on modern CPUs (recommended)"),
    ("distil-large-v3", "1.5 GB", "Best accuracy that still streams on a fast CPU"),
    ("large-v3-turbo", "1.6 GB", "Top accuracy, needs a GPU or batch mode"),
]

LANGUAGES = [
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
    ("pt", "Portuguese"),
    ("zh", "Chinese"),
    ("ja", "Japanese"),
    ("auto", "Auto-detect (slower)"),
]


_cached_model = None
_cached_model_name: str | None = None


def _get_or_load_model(model_name: str | None = None):
    """Get or load a whisper model. Caches in module global."""
    global _cached_model, _cached_model_name

    if model_name is None:
        from voiceio.config import load
        cfg = load()
        model_name = cfg.model.name

    if _cached_model is not None and _cached_model_name == model_name:
        return _cached_model

    from voiceio.worker import load_model
    # Cache-first, falling back to the network — which is what makes this the
    # download path too (_download_model calls here for a not-yet-cached model).
    _cached_model = load_model(model_name)
    _cached_model_name = model_name
    return _cached_model


def _download_model(model_name: str) -> bool:
    """Download the whisper model with a spinner."""
    print(f"  {DIM}This only happens once. The model is cached locally.{RESET}")

    with Spinner(f"Downloading model '{model_name}'...") as sp:
        try:
            # Suppress HuggingFace "unauthenticated requests" warning during download
            import logging
            hf_logger = logging.getLogger("huggingface_hub")
            prev_level = hf_logger.level
            hf_logger.setLevel(logging.ERROR)
            try:
                _get_or_load_model(model_name)
            finally:
                hf_logger.setLevel(prev_level)
            sp.ok(f"Model '{model_name}' ready!")
            return True
        except Exception as e:
            sp.fail(f"Download failed: {e}")
            return False


def _streaming_test(model=None, language: str | None = None) -> str:
    """Record audio and stream transcription results in real-time. Returns transcribed text."""
    import numpy as np
    import sounddevice as sd

    from voiceio.postprocess import cleanup

    sample_rate = 16000
    chunk_secs = 0.5
    chunk_size = int(sample_rate * chunk_secs)
    silence_threshold = 0.01
    max_duration = 10

    if model is None:
        with Spinner("Loading model...") as sp:
            model = _get_or_load_model()
            sp.ok("Model loaded")

    if language is None:
        from voiceio.config import load
        cfg = load()
        language = cfg.model.language

    lang = language if language != "auto" else None
    lang_str = lang or "en"

    print(f"\n  {YELLOW}Speak now!{RESET} (up to {max_duration}s, stops on 1.5s silence)")
    print(f"  {DIM}{'─' * 40}{RESET}")
    sys.stdout.write(f"  {BOLD}")
    sys.stdout.flush()

    audio_chunks: list[np.ndarray] = []
    silent_time = 0.0
    last_text = ""
    last_text_len = 0  # characters printed on screen
    recording = True

    def callback(indata, frames, time_info, status):
        nonlocal silent_time, recording
        if not recording:
            return
        audio_chunks.append(indata.copy())
        rms = np.sqrt(np.mean(indata ** 2))
        if rms < silence_threshold:
            silent_time += chunk_secs
        else:
            silent_time = 0.0

    stream = sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="float32",
        blocksize=chunk_size, callback=callback,
    )
    stream.start()
    start_time = time.time()

    try:
        while recording:
            time.sleep(0.8)
            elapsed = time.time() - start_time

            if elapsed > max_duration:
                recording = False
                break

            # Stop on sustained silence (but only after we've heard something)
            if silent_time >= 1.5 and len(audio_chunks) > 3:
                recording = False
                break

            if not audio_chunks:
                continue

            audio = np.concatenate(audio_chunks, axis=0).flatten()
            if len(audio) < sample_rate * 0.5:
                continue

            # Transcribe everything so far
            segments, _ = model.transcribe(audio, language=lang, beam_size=5, vad_filter=True)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            # Light cleanup for streaming preview (capitalization, punctuation spacing)
            text = cleanup(text, lang_str)

            if text and text != last_text:
                # Clear previous text and rewrite
                if last_text_len > 0:
                    sys.stdout.write("\b" * last_text_len + " " * last_text_len + "\b" * last_text_len)
                sys.stdout.write(text)
                sys.stdout.flush()
                last_text_len = len(text)
                last_text = text
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()

    # Final transcription with cleanup (capitalization, punctuation spacing)
    if audio_chunks:
        audio = np.concatenate(audio_chunks, axis=0).flatten()
        if len(audio) >= sample_rate * 0.3:
            segments, _ = model.transcribe(audio, language=lang, beam_size=5, vad_filter=True)
            final_text = " ".join(seg.text.strip() for seg in segments).strip()
            final_text = cleanup(final_text, lang_str)
            if final_text and final_text != last_text:
                if last_text_len > 0:
                    sys.stdout.write("\b" * last_text_len + " " * last_text_len + "\b" * last_text_len)
                sys.stdout.write(final_text)
                last_text = final_text

    sys.stdout.write(f"{RESET}\n")
    print(f"  {DIM}{'─' * 40}{RESET}")

    if last_text:
        print(f"  {GREEN}✓{RESET} Transcribed successfully!")
    else:
        print(f"  {YELLOW}⚠{RESET}  No speech detected. Check your microphone.")

    return last_text


def run_test() -> None:
    """Standalone test command: voiceio-test."""
    print(f"{CYAN}{BOLD}voiceio doctor --live{RESET}\n")

    options = [
        ("Mic + streaming transcription", "Test your microphone and see real-time transcription"),
        ("Hotkey test", "Verify your keyboard shortcut works"),
        ("Full test", "Both of the above"),
    ]

    idx = _ask_choice(options, default=0)

    if idx in (0, 2):
        print(f"\n{BOLD}Mic test{RESET}")
        print(f"{DIM}{'─' * 40}{RESET}")
        _streaming_test()

    if idx in (1, 2):
        print(f"\n{BOLD}Hotkey test{RESET}")
        print(f"{DIM}{'─' * 40}{RESET}")
        from voiceio.config import load
        cfg = load()
        ok = _test_hotkey(cfg.hotkey.key, cfg.hotkey.backend)
        if not ok:
            print(f"\n  {YELLOW}Troubleshooting:{RESET}")
            print(f"  {DIM}• On Wayland/GNOME: run {BOLD}voiceio setup{RESET}")
            print(f"  {DIM}• Or add shortcut manually: Settings → Keyboard → Custom Shortcuts{RESET}")
            print(f"  {DIM}  Command: voiceio toggle{RESET}")

    print()
