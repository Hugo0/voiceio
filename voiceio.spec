# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building voiceio as a standalone Windows executable.

Usage (from repo root):
    pip install pyinstaller
    pyinstaller voiceio.spec

Produces: dist/voiceio/voiceio.exe (one-dir mode for faster startup).
"""
import os
import sys
from pathlib import Path

block_cipher = None

# faster-whisper ships shared libs that PyInstaller doesn't auto-detect
# ctranslate2 has .dll/.so files we need to bundle
ct2_path = None
try:
    import ctranslate2
    ct2_path = Path(ctranslate2.__file__).parent
except ImportError:
    pass

# Collect all hidden imports that PyInstaller might miss
hidden_imports = [
    "voiceio",
    "voiceio.cli",
    "voiceio.app",
    "voiceio.config",
    "voiceio.feedback",
    "voiceio.health",
    "voiceio.pidlock",
    "voiceio.platform",
    "voiceio.recorder",
    "voiceio.service",
    "voiceio.streaming",
    "voiceio.transcriber",
    "voiceio.wizard",
    "voiceio.backends",
    "voiceio.hotkeys",
    "voiceio.hotkeys.chain",
    "voiceio.hotkeys.pynput_backend",
    "voiceio.hotkeys.socket_backend",
    "voiceio.typers",
    "voiceio.typers.chain",
    "voiceio.typers.clipboard",
    "voiceio.typers.pynput_type",
    "voiceio.tray",
    "voiceio.tray._icons",
    "voiceio.tray._pystray",
    # faster-whisper / ctranslate2 internals
    "faster_whisper",
    "ctranslate2",
    "huggingface_hub",
    # pynput backends
    "pynput.keyboard._win32",
    "pynput.mouse._win32",
    # sounddevice
    "sounddevice",
    "_sounddevice_data",
    # clipboard
    "pyperclip",
    # notifications
    "win11toast",
]

# Data files to include
datas = [
    # WAV sound files
    ("voiceio/sounds/*.wav", "voiceio/sounds"),
]

# Add ctranslate2 shared libraries if found
if ct2_path:
    for ext in ("*.dll", "*.so", "*.dylib", "*.pyd"):
        import glob
        libs = glob.glob(str(ct2_path / ext))
        if libs:
            datas.append((str(ct2_path / ext), "ctranslate2"))

a = Analysis(
    ["voiceio/cli.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Linux-only modules (not needed on Windows)
        "evdev",
        "gi",
        "fcntl",
        "termios",
        "tty",
        # IBus (Linux-only)
        "voiceio.ibus",
        "voiceio.ibus.engine",
        # Linux-only typers
        "voiceio.typers.ibus",
        "voiceio.typers.xdotool",
        "voiceio.typers.ydotool",
        "voiceio.typers.wtype",
        # Linux-only hotkey backend
        "voiceio.hotkeys.evdev",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="voiceio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=None,  # TODO: add voiceio.ico when available
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="voiceio",
)
