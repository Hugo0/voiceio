"""Input gain follows the recordings: down on clipping, up when very quiet,
never back to a level that clipped."""
from __future__ import annotations

from voiceio import mic_gain
from voiceio.mic_gain import MicGain


class FakeSource:
    def __init__(self, volume):
        self.volume = volume
        self.sets = []

    def get(self):
        return self.volume

    def set(self, v):
        self.sets.append(v)
        self.volume = v
        return True


def _gain(volume=0.5):
    g = MicGain(enabled=False)
    g._source = FakeSource(volume)
    return g


def test_clipping_lowers_by_a_fifth():
    g = _gain(0.5)
    change = g.after_recording(peak=1.0, clip_ratio=0.02, secs=10)
    assert change.reason == "clipping" and g._source.volume == 0.4


def test_never_below_the_floor():
    g = _gain(0.11)
    g.after_recording(1.0, 0.02, 10)
    assert g._source.volume == mic_gain.FLOOR


def test_three_quiet_recordings_raise_but_not_to_a_level_that_clipped():
    g = _gain(0.5)
    g.after_recording(1.0, 0.02, 10)                 # 0.5 clipped -> 0.4
    for _ in range(2):
        assert g.after_recording(0.05, 0.0, 10) is None
    assert g.after_recording(0.05, 0.0, 10) is None  # third quiet one: would reach 0.5
    assert g._source.volume == 0.4                  # capped at 0.5 * 0.8: no oscillation


def test_quiet_without_any_clipping_history_raises():
    g = _gain(0.3)
    for _ in range(3):
        change = g.after_recording(0.05, 0.0, 10)
    assert change.new == round(0.3 / mic_gain.STEP, 2)


def test_short_or_dead_recordings_dont_count():
    g = _gain(0.3)
    for _ in range(5):
        assert g.after_recording(0.05, 0.0, 1.0) is None   # too short
        assert g.after_recording(0.0, 0.0, 10) is None     # muted mic: not a gain problem
    assert g._source.sets == []


def test_normal_level_resets_the_quiet_streak():
    g = _gain(0.3)
    g.after_recording(0.05, 0.0, 10)
    g.after_recording(0.05, 0.0, 10)
    g.after_recording(0.5, 0.0, 10)
    assert g.after_recording(0.05, 0.0, 10) is None


def test_a_chosen_device_is_left_alone():
    assert not MicGain(enabled=True, device="hw:1").active


def test_off_without_volume_tools(monkeypatch):
    # conftest: quiet_audio._which finds nothing in tests
    assert not MicGain(enabled=True).active


def test_failures_never_raise():
    g = _gain(0.5)
    g._source.get = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    assert g.after_recording(1.0, 0.02, 10) is None
