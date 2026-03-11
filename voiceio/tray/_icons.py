"""Pre-render monochrome waveform tray icons as PNG files.

Idle: small static white waveform.
Recording: breathing white waveform (animation = the indicator).
Minimalist, symbolic-style to match GNOME/KDE panel aesthetics.

Icons are placed inside a freedesktop icon theme directory structure
so AppIndicator3 can resolve them by name.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

FRAME_COUNT = 12
ICON_SIZE = 256  # render 2x, DE downscale gives clean antialiasing
ANIM_INTERVAL_MS = 120  # ~8fps — smooth enough for wave travel


def _render_wave_png(
    size: int,
    color: tuple[int, int, int, int],
    amplitude: float = 1.0,
    phase_offset: float = 0.0,
) -> bytes:
    """Render a smooth sine wave line icon and return PNG bytes."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin_x = size * 0.15
    cy = size / 2
    max_amp = size * 0.3  # max vertical displacement from center
    stroke = max(size * 0.12, 3)  # thick enough to read at tray size
    periods = 1.5  # 1.5 sine periods across the icon

    # Build polyline points along a sine wave
    steps = 80
    points = []
    for i in range(steps + 1):
        t = i / steps
        x = margin_x + t * (size - 2 * margin_x)
        y = cy - amplitude * max_amp * math.sin(
            2 * math.pi * periods * t + phase_offset
        )
        points.append((x, y))

    # Draw with rounded joints for smoothness
    draw.line(points, fill=color, width=int(stroke), joint="curve")
    # Round the line caps by drawing circles at endpoints
    r = stroke / 2
    for px, py in (points[0], points[-1]):
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color)

    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_to_dir() -> tuple[Path, list[str]]:
    """Write all icon PNGs inside a freedesktop icon theme directory.

    Returns (theme_dir, [icon_names]) where icon_names[0] is "voiceio-idle"
    and icon_names[1..N] are "voiceio-rec-0" .. "voiceio-rec-7".
    theme_dir should be passed to AppIndicator set_icon_theme_path().
    """
    theme_dir = Path(tempfile.mkdtemp(prefix="voiceio-tray-"))
    apps_dir = theme_dir / "hicolor" / f"{ICON_SIZE}x{ICON_SIZE}" / "apps"
    apps_dir.mkdir(parents=True)

    # Write index.theme so GTK recognizes this as a valid icon theme
    index = theme_dir / "hicolor" / "index.theme"
    index.write_text(
        "[Icon Theme]\n"
        "Name=voiceio\n"
        f"Directories={ICON_SIZE}x{ICON_SIZE}/apps\n\n"
        f"[{ICON_SIZE}x{ICON_SIZE}/apps]\n"
        f"Size={ICON_SIZE}\n"
        "Type=Fixed\n"
    )

    white = (255, 255, 255, 220)
    names = []

    # Idle: gentle breathing + slow drift — alive but calm
    # ~6s full cycle at 480ms/frame (12 frames), full 2π phase = seamless loop
    for i in range(FRAME_COUNT):
        t = 2 * math.pi * i / FRAME_COUNT
        amp = 0.2 + 0.15 * math.sin(t)  # 0.2 → 0.35 → 0.2 (subtle)
        phase = t  # full 2π cycle — loops seamlessly

        name = f"voiceio-idle-{i}"
        (apps_dir / f"{name}.png").write_bytes(
            _render_wave_png(ICON_SIZE, white, amplitude=amp, phase_offset=phase)
        )
        names.append(name)

    # Recording: breathing amplitude + wave scrolls horizontally
    # Full 2π phase travel = one complete wave period of horizontal motion
    # Amplitude breathes between 0.4 and 1.0 (one breath cycle)
    for i in range(FRAME_COUNT):
        t = 2 * math.pi * i / FRAME_COUNT
        amp = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(t))  # 0.4 → 1.0 → 0.4
        phase = 2 * math.pi * i / FRAME_COUNT  # full cycle horizontal travel

        name = f"voiceio-rec-{i}"
        (apps_dir / f"{name}.png").write_bytes(
            _render_wave_png(ICON_SIZE, white, amplitude=amp, phase_offset=phase)
        )
        names.append(name)

    # Processing: fast constant-amplitude scroll (urgency/progress feel)
    # Constant high amplitude, smooth horizontal travel
    for i in range(FRAME_COUNT):
        phase = 2 * math.pi * i / FRAME_COUNT  # one full cycle scroll
        name = f"voiceio-proc-{i}"
        (apps_dir / f"{name}.png").write_bytes(
            _render_wave_png(ICON_SIZE, white, amplitude=0.7, phase_offset=phase)
        )
        names.append(name)

    return theme_dir, names
