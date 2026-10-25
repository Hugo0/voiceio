"""User presence: when may background learning use the computer?"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from voiceio import presence
from voiceio.presence import SeatSession, SessionProbe


@dataclass
class Probe:
    idle: float | None = None
    locked: bool = False
    inhibited: bool = False
    session: SeatSession | None = None

    def idle_seconds(self):
        return self.idle

    def screen_locked(self):
        return self.locked

    def idle_inhibited(self):
        return self.inhibited

    def seat_session(self):
        return self.session


AWAY = 600


@pytest.mark.parametrize("probe, here, why", [
    (Probe(idle=3), True, "last input 3 s ago"),
    (Probe(idle=3, locked=True), False, "screen locked"),          # locked, then walked off
    (Probe(idle=1800), False, "away 30 min"),
    (Probe(idle=1800, inhibited=True), True, "keeping the session awake"),   # a video
    (Probe(idle=1800, inhibited=True, locked=True), False, "away"),
    (Probe(idle=3 * 3600, inhibited=True), False, "away 180 min"),   # an agent, not a viewer
    # No idle monitor: logind decides, and "can't tell" means in use.
    (Probe(), True, "can't tell"),
    (Probe(session=SeatSession(exists=False)), False, "nobody is logged in"),
    (Probe(session=SeatSession(exists=True, locked=True)), False, "screen locked"),
    (Probe(session=SeatSession(exists=True, idle=True, idle_secs=900)), False, "away 15 min"),
    (Probe(session=SeatSession(exists=True, idle=True, idle_secs=60)), True, "in use"),
    (Probe(session=SeatSession(exists=True)), True, "in use"),
])
def test_presence_decision(probe, here, why):
    p = presence.check(AWAY, probe)
    assert p.here is here and why in p.why


def _bus(answers: dict[str, str]):
    """A busctl stand-in: the first answer whose key is in the argv wins;
    no answer is a failed call."""
    calls = []

    def call(*argv):
        calls.append(argv)
        line = " ".join(argv)
        return next((out for key, out in answers.items() if key in line), None)
    call.calls = calls
    return call


def test_probe_reads_mutter_then_kde_idle_time():
    assert SessionProbe(_bus({"Mutter": "t 2527\n"})).idle_seconds() == pytest.approx(2.527)
    assert SessionProbe(_bus({"GetSessionIdleTime": "u 30\n"})).idle_seconds() == 30.0
    assert SessionProbe(_bus({})).idle_seconds() is None


def test_probe_reads_lock_and_inhibitor_state():
    probe = SessionProbe(_bus({"org.gnome.ScreenSaver": "b true\n",
                               "IsInhibited": "b false\n"}))
    assert probe.screen_locked() and not probe.idle_inhibited()
    assert not SessionProbe(_bus({})).idle_inhibited()      # no session manager: not inhibited


def test_probe_reads_the_logind_graphical_session():
    bus = _bus({"Display": '(so) "3" "/org/freedesktop/login1/session/_33"\n',
                "IdleHint": "b true\nt 1000\nb false\n"})
    s = SessionProbe(bus).seat_session()
    assert s.exists and s.idle and not s.locked and s.idle_secs > 0
    assert "/org/freedesktop/login1/session/_33" in bus.calls[1]

    nobody = SessionProbe(_bus({"Display": '(so) "" "/"\n'})).seat_session()
    assert nobody == SeatSession(exists=False)
    assert SessionProbe(_bus({})).seat_session() is None
    # logind's lock flag stands in when GNOME's screensaver isn't there
    locked = _bus({"Display": '(so) "3" "/s/_33"\n', "IdleHint": "b false\nt 0\nb true\n"})
    assert SessionProbe(locked).screen_locked()


def test_fast_check_costs_two_calls_while_away():
    """Polled every 2 s during training: idle time and the inhibitor only."""
    bus = _bus({"Mutter": "t 3600000\n", "IsInhibited": "b false\n"})
    assert not presence.check(AWAY, SessionProbe(bus)).here
    assert len(bus.calls) == 2
