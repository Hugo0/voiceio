"""Interactive guided tour of voiceio features."""
from __future__ import annotations

import sys

from voiceio.wizard import (
    BOLD, CYAN, DIM, GREEN, RESET, YELLOW,
    Spinner, _get_or_load_model, _print_step, _rl_prompt, _streaming_test,
)


def _press_enter() -> None:
    try:
        input(_rl_prompt(f"\n  {DIM}Press Enter to continue...{RESET}"))
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def _countdown(label: str, secs: int = 3) -> None:
    """Visual countdown before a recording starts."""
    import time
    sys.stdout.write(f"\n  {YELLOW}{label}{RESET} ")
    sys.stdout.flush()
    for i in range(secs, 0, -1):
        sys.stdout.write(f"{BOLD}{i}{RESET} ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\n")
    sys.stdout.flush()


def run_demo() -> None:
    """Run the interactive demo tour."""
    from voiceio.config import load

    cfg = load()

    # Check if TTS is available
    has_tts = False
    tts_engine = None
    if cfg.tts.enabled:
        from voiceio.tts import select as tts_select
        tts_engine = tts_select(cfg.tts)
        has_tts = tts_engine is not None

    total_steps = 5 if has_tts else 4
    step_num = 0

    print(f"""
{CYAN}{BOLD}
 ██████╗ ███████╗███╗   ███╗ ██████╗
 ██╔══██╗██╔════╝████╗ ████║██╔═══██╗
 ██║  ██║█████╗  ██╔████╔██║██║   ██║
 ██║  ██║██╔══╝  ██║╚██╔╝██║██║   ██║
 ██████╔╝███████╗██║ ╚═╝ ██║╚██████╔╝
 ╚═════╝ ╚══════╝╚═╝     ╚═╝ ╚═════╝
{RESET}{DIM}  interactive guided tour{RESET}
""")

    print(f"  {DIM}This tour walks you through voiceio's main features.{RESET}")
    print(f"  {DIM}Each step demonstrates a capability with live audio.{RESET}")

    # ── Step 1: Basic Dictation ──────────────────────────────────────
    step_num += 1
    _print_step(step_num, total_steps, "Basic Dictation")
    print(f"  {DIM}voiceio turns your speech into text at your cursor.{RESET}")
    print(f"  {DIM}Let's try it — say a sentence into your microphone.{RESET}\n")

    with Spinner("Loading Whisper model...") as sp:
        model = _get_or_load_model(cfg.model.name)
        sp.ok("Model loaded")

    _countdown("Get ready to speak...")
    _streaming_test(model=model)

    print(f"\n  {DIM}In normal use, this text appears at your cursor in any app.{RESET}")
    print(f"  {DIM}Press {BOLD}{cfg.hotkey.key}{RESET}{DIM} to start/stop recording.{RESET}")
    _press_enter()

    # ── Step 2: Voice Commands ───────────────────────────────────────
    step_num += 1
    _print_step(step_num, total_steps, "Voice Commands")
    print(f"  {DIM}voiceio recognizes spoken commands mixed with dictation.{RESET}")
    print(f"  {DIM}Try saying something like:{RESET}\n")
    print(f"  {YELLOW}\"Hello comma new line how are you question mark\"{RESET}\n")

    commands_table = [
        ("new line / new paragraph", "insert line break"),
        ("period / comma / question mark", "insert punctuation"),
        ("scratch that", "delete last phrase"),
        ("correct that", "flag last word for review"),
        ("select all / copy that / undo", "editing commands"),
    ]
    print(f"  {BOLD}Available commands:{RESET}")
    for cmd, desc in commands_table:
        print(f"    {GREEN}{cmd}{RESET}  {DIM}— {desc}{RESET}")

    print(f"\n  {DIM}Now try it — say something with voice commands:{RESET}")
    _countdown("Get ready to speak...")
    _streaming_test(model=model)

    status = "enabled" if cfg.commands.enabled else "disabled"
    print(f"\n  {DIM}Voice commands are currently {BOLD}{status}{RESET}{DIM}. "
          f"Change in config or re-run 'voiceio setup'.{RESET}")
    _press_enter()

    # ── Step 3: Text-to-Speech (conditional) ─────────────────────────
    if has_tts:
        step_num += 1
        _print_step(step_num, total_steps, "Text-to-Speech")
        print(f"  {DIM}voiceio can also read text aloud. Select text and press{RESET}")
        print(f"  {BOLD}{cfg.tts.hotkey}{RESET}{DIM} to hear it spoken.{RESET}\n")
        print(f"  {DIM}Engine: {BOLD}{tts_engine.name}{RESET}")

        print(f"\n  {DIM}Let's play a demo sentence:{RESET}\n")
        demo_text = "Welcome to voiceio! Speech to text and text to speech, locally and instantly."

        with Spinner("Synthesizing...") as sp:
            try:
                from voiceio.tts.player import TTSPlayer
                audio, sample_rate = tts_engine.synthesize(
                    demo_text, cfg.tts.voice, cfg.tts.speed,
                )
                sp.ok("Playing audio")
                player = TTSPlayer()
                player.play(audio, sample_rate)
            except Exception as e:
                sp.fail(f"TTS failed: {e}")

        print(f"\n  {DIM}In normal use: select text → {BOLD}{cfg.tts.hotkey}{RESET}{DIM} → hear it spoken.{RESET}")
        print(f"  {DIM}Press the hotkey again to cancel playback.{RESET}")
        _press_enter()

    # ── Step N: Corrections ──────────────────────────────────────────
    step_num += 1
    _print_step(step_num, total_steps, "Corrections & Learning")
    print(f"  {DIM}voiceio learns from your dictation to fix recurring mistakes.{RESET}\n")

    corrections_table = [
        (f"{CYAN}voiceio correct \"wrong\" \"right\"{RESET}", "add a manual correction rule"),
        (f"{CYAN}voiceio correct --auto{RESET}", "scan history with LLM to find mistakes"),
        (f"{CYAN}voiceio correct --flagged{RESET}", "review words flagged with \"correct that\""),
        (f"{CYAN}voiceio correct --list{RESET}", "show all correction rules"),
    ]
    for cmd, desc in corrections_table:
        print(f"    {cmd}  {DIM}— {desc}{RESET}")

    print(f"\n  {DIM}During dictation, say {BOLD}\"correct that\"{RESET}{DIM} to flag a word.{RESET}")
    print(f"  {DIM}Corrections are applied to all future dictations automatically.{RESET}")
    _press_enter()

    # ── Step N: Cheat Sheet ──────────────────────────────────────────
    step_num += 1
    _print_step(step_num, total_steps, "What's Next")

    print(f"  {BOLD}Hotkeys:{RESET}")
    print(f"    {GREEN}{cfg.hotkey.key}{RESET}          toggle recording")
    if has_tts:
        print(f"    {GREEN}{cfg.tts.hotkey}{RESET}      read selected text aloud")

    print(f"\n  {BOLD}CLI commands:{RESET}")
    cli_commands = [
        ("voiceio", "start the daemon"),
        ("voiceio setup", "interactive setup wizard"),
        ("voiceio doctor", "system health check"),
        ("voiceio test", "mic + hotkey test"),
        ("voiceio correct", "manage corrections"),
        ("voiceio history", "view transcription history"),
        ("voiceio logs", "view daemon logs"),
        ("voiceio service", "manage autostart service"),
    ]
    for cmd, desc in cli_commands:
        print(f"    {CYAN}{cmd:<22}{RESET} {DIM}{desc}{RESET}")

    print(f"\n  {BOLD}Config:{RESET}")
    from voiceio.config import CONFIG_PATH
    print(f"    {DIM}{CONFIG_PATH}{RESET}")

    # Smart suggest: voiceio correct
    from voiceio.config import HISTORY_PATH
    history_count = 0
    if HISTORY_PATH.exists():
        try:
            lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()
            history_count = sum(1 for ln in lines if ln.strip())
        except OSError:
            pass

    print(f"\n{GREEN}{'━' * 50}{RESET}")
    print(f"{BOLD}  Tour complete!{RESET} You're ready to use voiceio.")
    print(f"  Press {BOLD}{cfg.hotkey.key}{RESET} in any app to start dictating.")
    if history_count >= 20:
        print(f"\n  {CYAN}Tip:{RESET} You have {history_count} dictation entries.")
        print(f"  Run {BOLD}voiceio correct{RESET} to scan for and fix Whisper mistakes.")
    else:
        print(f"\n  {DIM}After a few dictations, run {BOLD}voiceio correct{RESET}{DIM} to")
        print(f"  find and fix recurring Whisper mistakes.{RESET}")
    print(f"{GREEN}{'━' * 50}{RESET}\n")
