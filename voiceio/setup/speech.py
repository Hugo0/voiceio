"""Speech model choice, download, and the live mic/transcription test."""
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

# Offered (and preselected) only where it can run with vocabulary biasing:
# English, with a sherpa-onnx that has the NeMo TDT beam-search fix.
PARAKEET_CHOICE = ("parakeet-tdt-0.6b-v2", "480 MB",
                   "Fastest, best on names; English only (recommended)")
# The same model where the wizard can install that sherpa-onnx first
# (`voiceio models sherpa`'s installer). ~16 MB: sherpa-onnx + sherpa-onnx-core
# wheels for Linux x86_64 (2.0.3 on PyPI).
PARAKEET_INSTALL_CHOICE = (PARAKEET_CHOICE[0], PARAKEET_CHOICE[1],
                           "Fastest, best on names; English only "
                           "(recommended, installs sherpa-onnx ~16 MB)")


def model_choices(language: str = "en", *, offer_install: bool = False,
                  ) -> list[tuple[str, str, str]]:
    """The wizard's model menu, the default decoder first when it's Parakeet.

    `offer_install`: for English without a usable sherpa-onnx, still put
    Parakeet first, labelled as installing sherpa-onnx, where that install can
    work (see `fetch.can_install_sherpa`). Only an interactive wizard asks for
    it; the non-interactive setup never installs packages by itself.
    """
    from voiceio.core.decoders.catalog import default_model
    if default_model(language) == PARAKEET_CHOICE[0]:
        first = PARAKEET_CHOICE
    elif offer_install and language == "en" and _can_offer_sherpa_install():
        first = PARAKEET_INSTALL_CHOICE
    else:
        return MODELS
    return [first] + [
        (n, size, desc.replace(" (recommended)", "")) for n, size, desc in MODELS]


def _can_offer_sherpa_install() -> bool:
    """No sherpa-onnx at all, and an environment the installer can write to.
    An older sherpa-onnx already imported here can't be swapped for the new
    one inside this process: `voiceio models sherpa` upgrades that."""
    from voiceio.core.decoders.fetch import can_install_sherpa
    from voiceio.core.decoders.sherpa import features
    return not features().installed and can_install_sherpa()


def default_choice(choices: list[tuple[str, str, str]]) -> int:
    from voiceio.setup.ui import _index_of
    return 0 if choices[0][0] == PARAKEET_CHOICE[0] else _index_of(choices, "small", 2)


def _install_sherpa() -> bool:
    """Install sherpa-onnx for Parakeet, with a spinner; True once Parakeet is
    the default here. On any failure: one line, False (caller falls back)."""
    from voiceio.core.decoders import fetch
    from voiceio.core.decoders.catalog import default_model
    with Spinner(f"Installing {fetch.SHERPA_REQUIREMENT}...") as sp:
        try:
            fetch.install_sherpa(quiet=True)
        except OSError as e:
            sp.warn(f"Couldn't install sherpa-onnx ({e}); using Whisper small instead.")
            return False
        if default_model("en") != PARAKEET_CHOICE[0]:
            sp.warn("sherpa-onnx installed but can't load here; using Whisper small instead.")
            return False
        sp.ok("sherpa-onnx installed")
    return True


class _SherpaModel:
    """The wizard's mic test calls faster-whisper's `transcribe`; give a
    sherpa-onnx decoder that shape."""

    def __init__(self, decoder):
        self._decoder = decoder

    def transcribe(self, audio, **_):
        import types
        text = self._decoder.decode(audio, final=True).text
        return [types.SimpleNamespace(text=text)], None


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

    from voiceio.core.decoders.catalog import runtime_of
    if runtime_of(model_name) == "sherpa-onnx":
        from voiceio.config import ModelConfig
        from voiceio.core.decoders import build_decoder
        _cached_model = _SherpaModel(build_decoder(ModelConfig(name=model_name)))
        _cached_model_name = model_name
        return _cached_model

    from voiceio.core.decoders.faster_whisper.worker import load_model
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


# The mic test's timing. Silence ends the test only once something has been
# heard: an audio path with seconds of latency (remote desktop, IP-KVM,
# Bluetooth) delivers nothing but silence at first, and stopping then said
# "no speech" before the user's voice had even arrived.
_STOP_SILENCE_SECS = 1.5    # after speech: this much quiet ends the test
_WAIT_FOR_SPEECH_SECS = 12  # before speech: give up after this long
_MAX_SPEECH_SECS = 10       # after speech starts: record at most this long
# Voice first arriving later than this after "Speak now!" may be a delayed
# audio path rather than a slow start: suggest audio.input_latency_ms.
_LATE_VOICE_SECS = 1.8
# Typical time to start talking after the prompt; not part of the latency.
_REACTION_SECS = 0.6


def _test_should_stop(elapsed: float, heard_at: float | None, silent_secs: float) -> bool:
    """Whether the mic test is done, `elapsed` seconds in. `heard_at` is when
    speech was first heard (None: not yet); `silent_secs` is the quiet since."""
    if heard_at is None:
        return elapsed >= _WAIT_FOR_SPEECH_SECS
    return (silent_secs >= _STOP_SILENCE_SECS
            or elapsed - heard_at >= _MAX_SPEECH_SECS)


def _streaming_test(model=None, language: str | None = None) -> str:
    """Record audio and stream transcription results in real-time. Returns transcribed text."""
    import numpy as np
    import sounddevice as sd

    from voiceio.postprocess import cleanup

    sample_rate = 16000
    chunk_secs = 0.5
    chunk_size = int(sample_rate * chunk_secs)
    silence_threshold = 0.01

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

    print(f"\n  {YELLOW}Speak now!{RESET} (stops {_STOP_SILENCE_SECS:g}s after you finish, "
          f"waits up to {_WAIT_FOR_SPEECH_SECS}s for your voice)")
    print(f"  {DIM}{'─' * 40}{RESET}")
    sys.stdout.write(f"  {BOLD}")
    sys.stdout.flush()

    audio_chunks: list[np.ndarray] = []
    silent_time = 0.0       # seconds of audio below the threshold, in a row
    heard_sound = False     # a chunk above the threshold arrived
    samples_seen = 0
    onset: float | None = None  # where in the audio the first sound starts, seconds
    heard_at: float | None = None
    last_text = ""
    last_text_len = 0  # characters printed on screen
    recording = True

    def callback(indata, frames, time_info, status):
        nonlocal silent_time, heard_sound, samples_seen, onset
        if not recording:
            return
        audio_chunks.append(indata.copy())
        rms = np.sqrt(np.mean(indata ** 2))
        if rms < silence_threshold:
            silent_time += frames / sample_rate
        else:
            silent_time = 0.0
            heard_sound = True
            if onset is None:
                onset = (samples_seen + _first_loud_sample(
                    indata, silence_threshold, sample_rate)) / sample_rate
        samples_seen += frames

    stream = sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="float32",
        blocksize=chunk_size, callback=callback,
    )
    stream.start()
    start_time = time.time()
    elapsed = 0.0

    try:
        while recording:
            time.sleep(0.8)
            elapsed = time.time() - start_time

            if heard_at is None and heard_sound:
                heard_at = elapsed
            if _test_should_stop(elapsed, heard_at, silent_time):
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
                # New words are speech too, even from a mic too quiet to
                # cross the threshold: for that mic they are the only sign
                # of speech, so they also hold off the silence stop.
                if heard_at is None:
                    heard_at = time.time() - start_time
                if not heard_sound:
                    silent_time = 0.0
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
        recording = False
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
        print(f"  {YELLOW}⚠{RESET}  {_no_speech_message(elapsed, bool(audio_chunks), heard_sound)}")
    hint = _latency_hint(onset)
    if hint:
        print(f"  {DIM}{hint}{RESET}")

    return last_text


def _first_loud_sample(block, threshold: float, sample_rate: int) -> int:
    """Offset of the first 20 ms frame of `block` at or above `threshold` RMS
    (the block as a whole is above it)."""
    import numpy as np
    flat = block.reshape(-1)
    step = max(1, sample_rate // 50)
    for start in range(0, len(flat), step):
        frame = flat[start:start + step]
        if np.sqrt(np.mean(frame ** 2)) >= threshold:
            return start
    return 0


def _latency_hint(onset: float | None) -> str | None:
    """One line suggesting audio.input_latency_ms when the voice arrived late.
    `onset` is where the first sound is in the test's audio (None: none).
    A typical reaction time is taken off; the rest may still include a slow
    start, so it is worded as a suggestion, never set automatically."""
    from voiceio.config import MAX_INPUT_LATENCY_MS
    if onset is None or onset <= _LATE_VOICE_SECS:
        return None
    ms = min(max(int(round((onset - _REACTION_SECS) * 10)) * 100, 100), MAX_INPUT_LATENCY_MS)
    return (f"Your voice arrived {onset:.1f}s after the test began. If your audio goes "
            "through a remote desktop, KVM or Bluetooth link, keep your last words "
            f"with: voiceio config set audio.input_latency_ms {ms}")


def _no_speech_message(elapsed: float, got_audio: bool, heard_sound: bool) -> str:
    if not got_audio:
        return (f"The microphone delivered no audio in {elapsed:.0f}s. "
                "Check the input device in your sound settings.")
    if heard_sound:
        return ("Heard sound, but no words were recognized. "
                "Speak closer to the microphone or raise its input level.")
    return (f"No speech arrived within {elapsed:.0f}s. Check your microphone. "
            "If your audio goes through a remote desktop, IP-KVM or Bluetooth "
            "link with a delay, keep talking until your words appear.")


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
