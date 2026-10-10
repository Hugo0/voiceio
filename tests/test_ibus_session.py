"""IBusSession: engine lifecycle + recording-time input-source claim.

Everything here mocks subprocess — nothing may touch the real IBus/GNOME
session. History: three rounds of preedit-flash / dead-keyboard bugs came
from this lifecycle, so most tests pin a specific past failure.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from voiceio.ibus import session as sess
from voiceio.ibus.session import (
    GnomeInputSources,
    IBusEngineSources,
    IBusSession,
)

RUN = "voiceio.ibus.session.subprocess.run"


_spawned: list[threading.Thread] = []


class _TrackedThread(threading.Thread):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        _spawned.append(self)


@pytest.fixture(autouse=True)
def _no_real_ibus(monkeypatch):
    monkeypatch.setattr(sess, "_ibus_env", lambda: {})
    monkeypatch.setattr(sess.time, "sleep", lambda s: None)
    monkeypatch.setattr(sess.threading, "Thread", _TrackedThread)
    _spawned.clear()


class _Gen:
    def __init__(self, value=1):
        self.value = value

    def __call__(self):
        return self.value


def _session(*, gnome=True, idle=True, gen=None, sources=None):
    platform = MagicMock(is_gnome=gnome)
    state = {"idle": idle}
    gen = gen or _Gen()
    s = IBusSession(
        lambda: platform, is_idle=lambda: state["idle"],
        generation=gen, sources=sources,
    )
    return s, state, gen


def _join_threads(timeout=2.0):
    """Wait for the background claim/restore threads this test started."""
    deadline = time.monotonic() + timeout
    for t in list(_spawned):
        t.join(max(0.0, deadline - time.monotonic()))
        assert not t.is_alive(), "session thread hung"


def _cmds(run):
    return [c[0][0] for c in run.call_args_list]


# ── claim / release generations ────────────────────────────────────────

class TestClaimRelease:
    def _with_mock_sources(self, gen_value=1):
        sources = MagicMock()
        s, state, gen = _session(sources=sources, gen=_Gen(gen_value))
        s._proc = MagicMock()  # engine resident: no respawn thread
        return s, sources, gen

    def test_claim_activates_and_release_restores(self):
        s, sources, gen = self._with_mock_sources()
        s.claim(1)
        _join_threads()
        sources.activate_voiceio.assert_called_once()
        gen.value = 2  # stop bumps
        s.release(2)
        _join_threads()
        sources.restore.assert_called_once()

    def test_superseded_generation_does_not_restore(self):
        """A finalizer from recording A finishing after recording B started
        must not restore the source — it would yank it from under B."""
        s, sources, gen = self._with_mock_sources()
        s.claim(1)            # start A
        gen.value = 2         # stop A → finalizer A holds gen 2
        gen.value = 3         # start B
        s.claim(3)
        s.release(2)          # finalizer A finishes late
        _join_threads()
        sources.restore.assert_not_called()
        gen.value = 4         # stop B
        s.release(4)
        _join_threads()
        sources.restore.assert_called_once()

    def test_release_without_claim_is_noop(self):
        s, sources, gen = self._with_mock_sources()
        with patch("voiceio.ibus.session.threading.Thread") as MockThread:
            s.release(1)
        MockThread.assert_not_called()
        sources.restore.assert_not_called()

    def test_release_runs_off_thread(self):
        s, sources, gen = self._with_mock_sources()
        s._wanted = 1
        s._claimer = sources
        with patch("voiceio.ibus.session.threading.Thread") as MockThread:
            s.release(1)
        MockThread.assert_called_once()

    def test_release_before_activation_skips_claim(self):
        """Very short recording: release lands before the activation thread
        gets the source lock — the claim must not happen afterwards (it
        would leave voiceio active while idle)."""
        s, sources, gen = self._with_mock_sources()
        with s._source_lock:   # hold activation off
            s.claim(1)
            gen.value = 2
            s.release(2)
        _join_threads()
        sources.activate_voiceio.assert_not_called()

    def test_record_start_respawns_reaped_engine(self):
        sources = MagicMock()
        s, state, gen = _session(sources=sources, idle=False)
        spawned = threading.Event()
        with patch.object(s, "ensure_engine", side_effect=lambda: spawned.set()):
            s.claim(1)
            assert spawned.wait(timeout=2), "engine respawn not triggered"
        _join_threads()


# ── hijack ─────────────────────────────────────────────────────────────

class TestHijack:
    def test_hijack_reported_once_per_recording(self):
        sources = MagicMock()
        sources.switched_away.return_value = True
        s, _, _ = _session(sources=sources)
        s._wanted, s._claimer = 1, sources
        assert s.hijacked() is True
        assert s.hijacked() is False  # already fell back
        s._proc = MagicMock()
        s.claim(2)  # next recording re-arms
        _join_threads()
        assert s.hijacked() is True

    def test_no_hijack_when_still_on_voiceio(self):
        sources = MagicMock()
        sources.switched_away.return_value = False
        s, _, _ = _session(sources=sources)
        s._wanted, s._claimer = 1, sources
        assert s.hijacked() is False

    def test_no_hijack_without_claim(self):
        sources = MagicMock()
        sources.switched_away.return_value = True
        s, _, _ = _session(sources=sources)
        assert s.hijacked() is False

    def test_gnome_switched_away(self):
        g = GnomeInputSources(lambda: "xkb:us::eng")
        g.voiceio_index = 3
        g.current = lambda: 0
        assert g.switched_away() is True
        g.current = lambda: 3
        assert g.switched_away() is False

    def test_plain_switched_away(self):
        p = IBusEngineSources(lambda: "xkb:us::eng")
        p.claimed = True
        p.current = lambda: "xkb:de::eng"
        assert p.switched_away() is True
        p.current = lambda: "voiceio"
        assert p.switched_away() is False


# ── GNOME sources ──────────────────────────────────────────────────────

class TestGnomeSources:
    def test_restore_uses_recorded_prev_index(self):
        g = GnomeInputSources(lambda: "xkb:us::eng")
        g.prev_index, g.voiceio_index = 2, 4
        calls = []
        g.set_index = calls.append
        g.release_engine = MagicMock()
        g.restore()
        assert calls == [2]  # restores the user's source, not hardcoded 0
        assert g.prev_index is None and g.voiceio_index is None

    def test_restore_defaults_to_zero_when_unknown(self):
        g = GnomeInputSources(lambda: "xkb:us::eng")
        calls = []
        g.set_index = calls.append
        g.release_engine = MagicMock()
        g.restore()
        assert calls == [0]

    def test_restore_releases_engine(self):
        g = GnomeInputSources(lambda: "xkb:us::eng")
        g.set_index = MagicMock()
        g.release_engine = MagicMock()
        g.restore()
        g.release_engine.assert_called_once()

    def test_release_sets_user_engine_unconditionally(self):
        """Release must not query first: a stale reading during the async
        activation made it skip itself and leave voiceio globally active."""
        g = GnomeInputSources(lambda: "xkb:us::eng")
        with patch(RUN) as run:
            run.return_value = MagicMock(returncode=0)
            g.release_engine()
        assert _cmds(run) == [["ibus", "engine", "xkb:us::eng"]]

    def test_activation_claims_global_engine(self):
        """gsettings 'current' is legacy/no-op on newer GNOME — activation
        must also claim the global engine directly."""
        g = GnomeInputSources(lambda: "xkb:us::eng")
        with patch(RUN) as run:
            run.side_effect = [
                MagicMock(returncode=0, stdout="[('xkb', 'us'), ('ibus', 'voiceio')]\n"),
                MagicMock(returncode=0, stdout="0\n"),   # current (prev)
                MagicMock(returncode=0),                 # gsettings set current
                MagicMock(returncode=0),                 # ibus engine voiceio
            ]
            g.activate_voiceio()
        assert ["ibus", "engine", "voiceio"] in _cmds(run)
        assert g.prev_index == 0 and g.voiceio_index == 1

    def test_shutdown_without_claim_touches_nothing(self):
        g = GnomeInputSources(lambda: "xkb:us::eng")
        with patch(RUN) as run:
            g.shutdown()
        run.assert_not_called()

    def test_shutdown_does_not_force_index_zero(self):
        """Claimed but prev unknown (user was already on voiceio, or the read
        failed): shutdown must not invent index 0."""
        g = GnomeInputSources(lambda: "xkb:us::eng")
        g.voiceio_index = 1  # claimed, prev_index None
        with patch(RUN) as run:
            g.shutdown()
        assert not any(c[:3] == ["gsettings", "set", sess._GSCHEMA] for c in _cmds(run))

    def test_shutdown_restores_claimed_prev(self):
        g = GnomeInputSources(lambda: "xkb:us::eng")
        g.prev_index, g.voiceio_index = 2, 1
        with patch(RUN) as run:
            g.shutdown()
        cmds = _cmds(run)
        assert ["gsettings", "set", sess._GSCHEMA, "current", "2"] in cmds
        assert ["ibus", "engine", "xkb:us::eng"] in cmds


# ── non-GNOME (KDE, sway) sources — the known-bug fix ──────────────────

class TestIBusEngineSources:
    def test_activate_switches_global_engine_and_remembers_prev(self):
        """Regression: on KDE/sway activation returned early, the engine
        never got focus, and every commit degraded to clipboard+paste."""
        p = IBusEngineSources(lambda: "xkb:us::eng")
        with patch(RUN) as run:
            run.side_effect = [
                MagicMock(returncode=0, stdout="xkb:de::eng\n"),  # ibus engine
                MagicMock(returncode=0),                          # ibus engine voiceio
            ]
            p.activate_voiceio()
        assert _cmds(run) == [["ibus", "engine"], ["ibus", "engine", "voiceio"]]
        assert p.claimed and p.prev_engine == "xkb:de::eng"
        assert not any("gsettings" in c[0] for c in _cmds(run))

    def test_restore_switches_back_to_prev(self):
        p = IBusEngineSources(lambda: "xkb:us::eng")
        p.claimed, p.prev_engine = True, "xkb:de::eng"
        with patch(RUN) as run:
            p.restore()
        assert _cmds(run) == [["ibus", "engine", "xkb:de::eng"]]
        assert not p.claimed and p.prev_engine is None

    def test_restore_uses_fallback_when_prev_unknown(self):
        p = IBusEngineSources(lambda: "xkb:fr::eng")
        p.claimed = True
        with patch(RUN) as run:
            p.restore()
        assert _cmds(run) == [["ibus", "engine", "xkb:fr::eng"]]

    def test_prev_voiceio_is_never_restored(self):
        p = IBusEngineSources(lambda: "xkb:us::eng")
        with patch(RUN) as run:
            run.side_effect = [
                MagicMock(returncode=0, stdout="voiceio\n"),
                MagicMock(returncode=0),
            ]
            p.activate_voiceio()
        assert p.prev_engine is None
        with patch(RUN) as run:
            p.restore()
        assert _cmds(run) == [["ibus", "engine", "xkb:us::eng"]]

    def test_session_picks_plain_sources_off_gnome(self):
        s, _, gen = _session(gnome=False)
        s._proc = MagicMock()
        with patch(RUN) as run:
            run.return_value = MagicMock(returncode=0, stdout="xkb:de::eng\n")
            s.claim(1)
            _join_threads()
            gen.value = 2
            s.release(2)
            _join_threads()
        cmds = _cmds(run)
        assert ["ibus", "engine", "voiceio"] in cmds
        assert cmds[-1] == ["ibus", "engine", "xkb:de::eng"]
        assert not any("gsettings" in c[0] for c in cmds)

    def test_shutdown_without_claim_touches_nothing(self):
        p = IBusEngineSources(lambda: "xkb:us::eng")
        with patch(RUN) as run:
            p.shutdown()
        run.assert_not_called()


# ── engine fallback ────────────────────────────────────────────────────

class TestEngineFallback:
    def test_fallback_derived_from_sources_when_prev_stale(self):
        """A stale 'voiceio' prev (inherited from a crashed daemon) must not
        be restored; derive the user's xkb layout instead."""
        s, _, _ = _session()
        s._prev_engine = "voiceio"
        with patch(RUN) as run:
            run.return_value = MagicMock(
                returncode=0, stdout="[('xkb', 'de'), ('ibus', 'voiceio')]\n",
            )
            assert s.engine_fallback() == "xkb:de::eng"

    def test_fallback_prefers_saved_prev(self):
        s, _, _ = _session()
        s._prev_engine = "xkb:fr::eng"
        with patch(RUN) as run:
            assert s.engine_fallback() == "xkb:fr::eng"
        run.assert_not_called()


# ── engine process lifecycle (health watchdog) ─────────────────────────

class TestEngineLifecycle:
    """Dormant-engine reaping is normal; recovery must not steal the source."""

    def _with_proc(self, proc_poll=-15):
        s, state, gen = _session(sources=MagicMock())
        proc = MagicMock()
        proc.poll.return_value = proc_poll
        proc.returncode = -15
        s._proc = proc
        return s, proc

    def test_idle_engine_death_respawns_without_activation(self):
        """ibus-daemon reaping a dormant engine must respawn it registered
        but NEVER activated — the activate/dormant dance grabbed the input
        source every cycle, and leaving the engine dead let ibus-daemon
        exec-spawn a socket-stealing duplicate on accidental activation."""
        s, proc = self._with_proc(proc_poll=-15)
        with patch.object(s, "ensure_engine") as ensure:
            s.health_tick(idle=True)
        ensure.assert_called_once_with(activate=False)
        assert not s.has_engine

    def test_mid_recording_engine_death_recovered(self):
        s, proc = self._with_proc(proc_poll=-15)
        with patch.object(s, "ensure_engine") as ensure:
            s.health_tick(idle=False)
        ensure.assert_called_once_with()  # default activate=True

    def test_single_missed_ping_tolerated(self):
        """One missed ping on a loaded system is not a zombie."""
        s, proc = self._with_proc(proc_poll=None)
        with patch.object(s, "ping", return_value=False), \
             patch.object(s, "ensure_engine") as ensure:
            s.health_tick(idle=True)
        ensure.assert_not_called()
        proc.kill.assert_not_called()
        assert s._ping_fails == 1

    def test_slow_starting_engine_is_not_a_zombie(self):
        """Under load the engine took 26s to open its socket; reaping it as a
        zombie restarted the slow start every 25s, in a loop."""
        s, proc = self._with_proc(proc_poll=None)
        s._started_at = time.monotonic() - 30
        with patch("voiceio.ibus.SOCKET_PATH") as sock, \
             patch.object(s, "ping", return_value=False), \
             patch.object(s, "ensure_engine") as ensure:
            sock.exists.return_value = False
            for _ in range(3):
                s.health_tick(idle=True)
        ensure.assert_not_called()
        proc.kill.assert_not_called()

    def test_two_missed_pings_restart_zombie(self):
        s, proc = self._with_proc(proc_poll=None)
        with patch.object(s, "ping", return_value=False), \
             patch.object(s, "ensure_engine") as ensure:
            s.health_tick(idle=True)
            s.health_tick(idle=True)
        ensure.assert_called_once_with(activate=False)
        proc.kill.assert_called_once()

    def test_successful_ping_resets_counter(self):
        s, proc = self._with_proc(proc_poll=None)
        s._ping_fails = 1
        with patch.object(s, "ping", return_value=True), \
             patch.object(s, "ensure_engine") as ensure:
            s.health_tick(idle=True)
        ensure.assert_not_called()
        assert s._ping_fails == 0

    def test_idle_tick_never_touches_input_source(self):
        """A responsive engine while idle => no source switching at all."""
        sources = MagicMock()
        s, _, _ = _session(sources=sources)
        s._proc = MagicMock()
        s._proc.poll.return_value = None
        with patch.object(s, "ping", return_value=True):
            s.health_tick(idle=True)
        assert sources.method_calls == []

    def test_no_ping_while_recording(self):
        s, proc = self._with_proc(proc_poll=None)
        with patch.object(s, "ping") as ping:
            s.health_tick(idle=False)
        ping.assert_not_called()

    def test_resume_forgets_dead_engine_without_respawn(self):
        s, proc = self._with_proc(proc_poll=-15)
        with patch.object(s, "ensure_engine") as ensure:
            s.on_resume()
        ensure.assert_not_called()
        assert not s.has_engine

    def test_dormant_spawn_never_touches_input_source(self):
        """activate=False must not run `ibus engine voiceio` or gsettings —
        idle spawns that claimed the source were the focus churn that wiped
        visible preedits and stole the keyboard mid-typing."""
        s, _, _ = _session()
        with patch(RUN) as run, \
             patch("voiceio.ibus.session.subprocess.Popen") as popen, \
             patch("voiceio.ibus.session.kill_stale_engines"), \
             patch("voiceio.ibus.SOCKET_PATH") as sock_path, \
             patch("voiceio.ibus.READY_PATH"):
            sock_path.exists.return_value = True
            run.return_value = MagicMock(returncode=0, stdout="xkb:us::eng\n")
            s.ensure_engine(activate=False)
        assert s.has_engine  # engine process was spawned
        cmds = _cmds(run) + [c[0][0] for c in popen.call_args_list if c[0]]
        assert ["ibus", "engine", "voiceio"] not in cmds
        assert not any("gsettings" in cmd[0] for cmd in cmds)

    def test_activating_spawn_stays_active_while_recording(self):
        """A spawn dance finishing during a live recording must not return
        the source to dormancy (the 'underlined text never appears' bug)."""
        sources = MagicMock()
        s, state, _ = _session(sources=sources, idle=False)
        with patch(RUN) as run, \
             patch("voiceio.ibus.session.subprocess.Popen"), \
             patch("voiceio.ibus.session.kill_stale_engines"), \
             patch("voiceio.ibus.SOCKET_PATH") as sock_path, \
             patch("voiceio.ibus.READY_PATH") as ready:
            sock_path.exists.return_value = True
            ready.exists.return_value = True
            run.return_value = MagicMock(returncode=0, stdout="xkb:us::eng\n")
            s.ensure_engine(activate=True)
            sources.return_to.assert_not_called()
            state["idle"] = True
            s.ensure_engine(activate=True)
            sources.return_to.assert_called_once()

    def test_spawn_is_single_flight(self):
        s, _, _ = _session()
        with patch.object(s, "_spawn") as spawn:
            with s._spawn_lock:
                s.ensure_engine(activate=False)
        spawn.assert_not_called()


# ── shutdown ───────────────────────────────────────────────────────────

class TestShutdown:
    def test_shutdown_does_not_force_index_zero(self):
        """Idle shutdown (nothing claimed) must leave the user's source alone
        — never `gsettings set ... current 0`."""
        s, _, _ = _session(gnome=True)
        with patch(RUN) as run, patch("voiceio.ibus.SOCKET_PATH"):
            run.return_value = MagicMock(returncode=0, stdout="xkb:de::eng\n")
            s.shutdown()
        assert not any(c[:2] == ["gsettings", "set"] for c in _cmds(run))
        # Global engine isn't ours → not touched either.
        assert not any(c[:2] == ["ibus", "engine"] and len(c) == 3 for c in _cmds(run))

    def test_shutdown_restores_live_claim(self):
        sources = MagicMock()
        s, _, _ = _session(sources=sources)
        s._wanted, s._claimer = 1, sources
        with patch(RUN) as run, patch("voiceio.ibus.SOCKET_PATH"):
            run.return_value = MagicMock(returncode=0, stdout="xkb:us::eng\n")
            s.shutdown()
        sources.shutdown.assert_called_once()

    def test_shutdown_terminates_engine(self):
        s, _, _ = _session()
        proc = MagicMock()
        s._proc = proc
        with patch(RUN) as run, patch("voiceio.ibus.SOCKET_PATH"):
            run.return_value = MagicMock(returncode=0, stdout="xkb:us::eng\n")
            s.shutdown()
        proc.terminate.assert_called_once()
        assert not s.has_engine

    def test_shutdown_releases_stranded_global_engine(self):
        """If ibus still points at voiceio, don't leave it on a dead engine."""
        s, _, _ = _session()
        s._prev_engine = "xkb:us::eng"
        with patch(RUN) as run, patch("voiceio.ibus.SOCKET_PATH"):
            run.return_value = MagicMock(returncode=0, stdout="voiceio\n")
            s.shutdown()
        assert ["ibus", "engine", "xkb:us::eng"] in _cmds(run)
