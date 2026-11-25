"""audio.input_latency_ms: stopping waits out the audio still on its way."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from test_hotkey_modes import _after_debounce, _make_app, _wait_stopped

from voiceio.app import _State
from voiceio.config import AudioConfig, load


@pytest.fixture
def tray():
    with patch("voiceio.app.tray") as t:
        yield t


def _app(mode="toggle", latency_ms=0, tail_ms=0):
    """Hotkey-level app (start/stop stubbed) with the stop feedback observable."""
    vio = _make_app(mode, tail_ms=tail_ms)
    vio.cfg.audio.input_latency_ms = latency_ms
    vio._play_record_cue = MagicMock()
    vio._quiet = MagicMock()
    vio._quiet.stop.side_effect = lambda then=None: then and then()
    vio.recorder = MagicMock()
    vio._typers = MagicMock()
    vio._spool = MagicMock()
    return vio


def _cues(vio):
    return [c.kwargs["start"] for c in vio._play_record_cue.call_args_list]


def _start_then_stop(vio):
    vio.on_hotkey()
    _after_debounce(vio)
    vio.on_hotkey()


def test_toggle_without_latency_stops_at_once(tray):
    vio = _app("toggle", latency_ms=0)
    _start_then_stop(vio)
    assert vio.stops == [1] and vio._pending_stop is None


def test_toggle_with_latency_records_the_tail(tray):
    vio = _app("toggle", latency_ms=120)
    vio.on_hotkey()
    _after_debounce(vio)
    pressed = time.monotonic()
    vio.on_hotkey()
    assert vio.stops == [] and vio._state == _State.RECORDING   # still capturing
    # Feedback at the press, not when the tail ends.
    assert _cues(vio) == [False]
    tray.set_recording.assert_called_with(False)
    tray.set_processing.assert_called_with(True)
    assert _wait_stopped(vio)
    assert time.monotonic() - pressed >= 0.11
    assert vio.stops == [1] and vio._pending_stop is None


def test_auto_tap_stop_gets_the_latency_tail(tray):
    vio = _app("auto", latency_ms=90, tail_ms=250)
    vio.on_hotkey_press()
    vio.on_hotkey_release()                  # a tap: latched
    _after_debounce(vio)
    vio.on_hotkey_press()                    # the stop tap
    assert vio._pending_stop.timer.interval == pytest.approx(0.09)
    assert _wait_stopped(vio)


@pytest.mark.parametrize("latency_ms,announced", [(0, False), (40, True)])
def test_hold_tail_is_release_tail_plus_latency(tray, latency_ms, announced):
    vio = _app("hold", latency_ms=latency_ms, tail_ms=60)
    vio.on_hotkey_press()
    vio.on_hotkey_release()
    assert vio._pending_stop.timer.interval == pytest.approx((60 + latency_ms) / 1000)
    # Without latency the stop chime comes with the stop, as before.
    assert _cues(vio) == ([False] if announced else [])
    assert _wait_stopped(vio)
    assert vio.stops == [1]


def test_toggle_press_during_tail_keeps_recording(tray):
    vio = _app("toggle", latency_ms=2000)
    _start_then_stop(vio)
    timer = vio._pending_stop.timer
    _after_debounce(vio)                     # a press inside the 0.8 s debounce is the same press
    vio.on_hotkey()                          # "I'm not done"
    assert vio._pending_stop is None and timer.finished.is_set()
    assert vio._state == _State.RECORDING and vio._recording == 1
    assert _cues(vio) == [False, True]       # stop chime, then the start chime again
    tray.set_recording.assert_called_with(True)
    tray.set_processing.assert_called_with(False)
    vio.recorder.set_on_auto_stop.assert_called_with(vio._on_auto_stop)
    # The next stop press stops the same recording, after its own tail.
    vio.cfg.audio.input_latency_ms = 50
    _after_debounce(vio)
    vio.on_hotkey()
    assert _wait_stopped(vio)
    assert vio.stops == [1]


def test_hold_press_during_tail_is_push_to_talk_again(tray):
    vio = _app("hold", latency_ms=100, tail_ms=0)
    vio.on_hotkey_press()
    vio.on_hotkey_release()
    _after_debounce(vio)
    vio.on_hotkey_press()
    assert vio._held is not None and vio._held[1] == 1
    time.sleep(0.15)
    assert vio.stops == []
    vio.on_hotkey_release()
    assert _wait_stopped(vio)
    assert vio.stops == [1]


def test_stale_tail_timer_never_stops_a_newer_recording(tray):
    vio = _app("toggle", latency_ms=80)
    _start_then_stop(vio)
    pending = vio._pending_stop
    vio._do_stop()                           # recording 1 ends some other way
    vio._do_start()                          # recording 2 within the tail
    time.sleep(0.15)
    assert vio.stops == [1] and vio._state == _State.RECORDING
    vio._stop_due(pending)                   # fired late: still nothing
    assert vio.stops == [1]


def test_withdrawn_stop_timer_does_nothing(tray):
    vio = _app("toggle", latency_ms=2000)
    _start_then_stop(vio)
    pending = vio._pending_stop
    _after_debounce(vio)
    vio.on_hotkey()                          # withdrawn
    vio._stop_due(pending)                   # as if the cancel lost the race
    assert vio.stops == [] and vio._state == _State.RECORDING


def test_cancel_during_tail_cancels_at_once(tray):
    vio = _app("toggle", latency_ms=2000)
    _start_then_stop(vio)
    timer = vio._pending_stop.timer
    vio.cancel()
    assert vio._state == _State.IDLE and vio._pending_stop is None
    assert timer.finished.is_set()   # timer cancelled
    vio.recorder.stop.assert_called_once()   # audio dropped, no finalize
    assert vio.stops == []
    assert _cues(vio) == [False]             # one stop chime, not two
    vio._quiet.stop.assert_called_once()
    tray.set_processing.assert_called_with(False)


def test_auto_stop_during_tail_leaves_it_to_the_tail(tray):
    vio = _app("toggle", latency_ms=100)
    _start_then_stop(vio)
    _after_debounce(vio)
    vio._request_stop()                      # silence auto-stop meanwhile
    assert vio.stops == []
    assert _wait_stopped(vio)
    assert vio.stops == [1]


def test_auto_stop_without_a_tail_stops_at_once(tray):
    vio = _app("toggle", latency_ms=2000)
    vio.on_hotkey()
    _after_debounce(vio)
    vio._request_stop()
    assert vio.stops == [1]


# ── config ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("given,used", [
    (0, 0), (2500, 2500), (5000, 5000), (-5, 0), (9000, 5000),
    ("2 s", 0), (True, 0), (300.7, 300),
])
def test_input_latency_is_validated(given, used):
    assert AudioConfig(input_latency_ms=given).input_latency_ms == used


def test_input_latency_from_the_config_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[audio]\ninput_latency_ms = 2000\n")
    assert load(path).audio.input_latency_ms == 2000
    path.write_text("[audio]\ninput_latency_ms = 60000\n")
    assert load(path).audio.input_latency_ms == 5000
    assert load(tmp_path / "missing.toml").audio.input_latency_ms == 0


def test_real_stop_after_the_tail_does_not_chime_twice(tray):
    from test_cancel import _make_app as _full_app
    vio = _full_app(streaming=False)
    vio.cfg.audio.input_latency_ms = 60
    vio._play_record_cue = MagicMock()
    vio._quiet.stop.side_effect = lambda then=None: then and then()
    vio._state, vio._recording = _State.RECORDING, 3
    with patch.object(vio, "_output_thread") as spawn:
        vio.on_hotkey()                      # stop press: chime now
        assert vio._state == _State.RECORDING
        deadline = time.monotonic() + 2
        while vio._state != _State.IDLE and time.monotonic() < deadline:
            time.sleep(0.005)
    assert vio._state == _State.IDLE
    spawn.assert_called_once()               # the tail's audio is decoded
    vio.recorder.stop.assert_called_once()
    assert _cues(vio) == [False]
    vio._quiet.stop.assert_called_once()
    assert vio._stop_announced == 0
