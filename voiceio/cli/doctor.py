"""`voiceio doctor` — check (and with --fix, repair) every backend."""
from __future__ import annotations

import argparse
import sys


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("doctor", help="Check that everything works")
    p.add_argument("--fix", action="store_true", help="Try to fix what is broken")
    p.add_argument("--live", action="store_true",
                   help="Also record from the mic and transcribe it")
    p.set_defaults(func=_cmd_doctor)


def _is_privileged_cmd(cmd: list[str]) -> bool:
    """True if a fix command runs sudo or pipes a remote install script."""
    joined = " ".join(cmd)
    return "sudo" in cmd or "sudo " in joined or "curl" in joined or "| sh" in joined


def _confirm_privileged(description: str, command_str: str) -> bool:
    """Show the exact privileged command and require an explicit y/N (default N).

    Used for every sudo / remote-script action in `doctor --fix` so nothing
    touches the system or runs as root without the user seeing and approving
    the exact command first.
    """
    print(f"\n  {description} needs to run:")
    print(f"    {command_str}")
    try:
        answer = input("  Run this now? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    return answer in ("y", "yes")


def _cmd_doctor(args: argparse.Namespace) -> None:
    """Run diagnostic health check, offer to fix issues."""
    if not sys.platform.startswith("linux"):
        print("Note: voiceio targets Linux; other platforms are best-effort.\n",
              file=sys.stderr)
    from voiceio.health import check_health, format_report, format_usage, usage_findings
    from voiceio import config as _config
    report = check_health()
    print(format_report(report))
    print(format_usage(usage_findings()))
    if args.live:
        from voiceio.setup import run_test
        run_test()

    # Permission probe: nothing voiceio persists should be group/world-readable.
    perm_issues = _config.check_permissions()
    if perm_issues:
        print("\nInsecure file permissions (should be 0600 files / 0700 dirs):")
        for path, mode, expected in perm_issues:
            print(f"  {oct(mode)[2:]:>4} → {oct(expected)[2:]:>4}  {path}")

    # System-dependency preflight: the #1 install funnel killer.
    from voiceio import platform as plat
    missing_deps = plat.check_system_deps()
    dep_cmd = plat.system_deps_install_cmd(missing_deps) if missing_deps else ""
    if missing_deps:
        print("\nMissing system dependencies:")
        for key in missing_deps:
            print(f"  ✗ {plat.dep_label(key)}")
        print(f"\nInstall them all with:\n  {dep_cmd}")

    fixable = [b for b in report.hotkey_backends + report.typer_backends
               if not b.ok and b.fix_cmd]

    if not args.fix:
        if fixable or missing_deps:
            names = ", ".join(b.name for b in fixable)
            if missing_deps:
                names = (names + ", " if names else "") + "system packages"
            print(f"\nRun 'voiceio doctor --fix' to auto-fix: {names}")
        if perm_issues:
            print("Run 'voiceio doctor --fix' to tighten file permissions.")
        from voiceio.hints import hint
        hint("vocab_suggest", "Run 'voiceio vocab suggest' to find names it keeps mishearing")
        sys.exit(0 if (report.all_ok and not perm_issues and not missing_deps) else 1)

    # Auto-fix mode
    print("\nAttempting fixes...\n")
    import subprocess

    if missing_deps and dep_cmd:
        print(f"  Fixing system packages: {dep_cmd}")
        try:
            subprocess.run(dep_cmd.split(), check=True, timeout=600)
            print("  Done.")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"  Failed: {e} — run the command above manually.")

    fixed_any = False

    if perm_issues:
        changed = _config.harden_permissions()
        print(f"  Tightened permissions on {changed} path(s).")
        fixed_any = fixed_any or changed > 0

    for b in report.hotkey_backends + report.typer_backends:
        if not b.ok and b.fix_cmd:
            cmd_str = " ".join(b.fix_cmd)
            if _is_privileged_cmd(b.fix_cmd):
                if not _confirm_privileged(f"Fix {b.name}", cmd_str):
                    print(f"  Skipped {b.name} (declined).")
                    continue
            else:
                print(f"  Fixing {b.name}: {cmd_str}")
            try:
                subprocess.run(b.fix_cmd, check=True, timeout=30)
                print("  Done.")
                fixed_any = True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"  Failed: {e}")

    # CLI symlinks
    if not report.cli_in_path:
        print("  Fixing CLI: creating symlinks in ~/.local/bin/")
        try:
            from voiceio.service import install_symlinks
            linked = install_symlinks()
            if linked:
                print(f"  Done, linked: {', '.join(linked)}")
                fixed_any = True
            else:
                print("  Failed: no scripts found to link")
        except Exception as e:
            print(f"  Failed: {e}")

    # IBus-specific fixes
    ibus_broken = [b for b in report.typer_backends
                   if b.name == "ibus" and not b.ok]
    for b in ibus_broken:
        if "input sources" in b.reason.lower():
            print(f"  Fixing {b.name}: adding VoiceIO to GNOME input sources")
            try:
                from voiceio.typers.ibus import _ensure_gnome_input_source
                _ensure_gnome_input_source()
                print("  Done.")
                fixed_any = True
            except Exception as e:
                print(f"  Failed: {e}")
        elif "component" in b.reason.lower():
            print(f"  Fixing {b.name}: installing IBus component")
            try:
                from voiceio.typers.ibus import install_component
                install_component()
                print("  Done.")
                fixed_any = True
            except Exception as e:
                print(f"  Failed: {e}")

    # LLM fixes
    from voiceio.config import load as _load_cfg
    _cfg = _load_cfg()
    if _cfg.llm.enabled:
        from voiceio.llm import OllamaStatus, diagnose_ollama, install_ollama, start_ollama, pull_model
        llm_status, _ = diagnose_ollama(_cfg.llm)
        if llm_status == OllamaStatus.NOT_INSTALLED:
            if sys.platform == "linux":
                if _confirm_privileged(
                    "Install Ollama (local LLM runtime)",
                    "curl -fsSL https://ollama.com/install.sh | sh",
                ):
                    print("  Installing Ollama...")
                    if install_ollama():
                        print("  Done.")
                        fixed_any = True
                    else:
                        print("  Failed. Visit https://ollama.com")
                else:
                    print("  Skipped Ollama install (declined). Manual: https://ollama.com/download")
            else:
                print("  LLM: Install Ollama from https://ollama.com")
        elif llm_status == OllamaStatus.NOT_RUNNING:
            print("  Fixing LLM: starting Ollama...")
            if start_ollama():
                print("  Done.")
                fixed_any = True
            else:
                print("  Failed. Try: ollama serve")
        elif llm_status == OllamaStatus.MODEL_NOT_FOUND:
            model = _cfg.llm.model or "phi3:mini"
            print(f"  Fixing LLM: pulling model '{model}'...")
            if pull_model(model):
                print("  Done.")
                fixed_any = True
            else:
                print(f"  Failed. Try: ollama pull {model}")

    if fixed_any:
        print("\nRe-checking...")
        report = check_health()
        print(format_report(report))

    sys.exit(0 if report.all_ok else 1)
