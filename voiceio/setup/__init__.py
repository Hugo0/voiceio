"""Interactive and non-interactive setup for voiceio.

``steps`` holds the interactive wizard (one function per step over a shared
``SetupState``); the other modules are the pieces those steps are built from.
"""
from voiceio.setup.noninteractive import run_setup_noninteractive
from voiceio.setup.speech import run_test
from voiceio.setup.steps import run_wizard

__all__ = ["run_setup_noninteractive", "run_test", "run_wizard"]
