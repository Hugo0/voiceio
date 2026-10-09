"""Terminal styling shared by the CLI, doctor, wizard and demo.

Honours NO_COLOR (https://no-color.org).
"""
from __future__ import annotations

import os

_on = not os.environ.get("NO_COLOR")

BOLD = "\033[1m" if _on else ""
DIM = "\033[2m" if _on else ""
RED = "\033[31m" if _on else ""
GREEN = "\033[32m" if _on else ""
YELLOW = "\033[33m" if _on else ""
CYAN = "\033[36m" if _on else ""
MAGENTA = "\033[35m" if _on else ""
RESET = "\033[0m" if _on else ""
