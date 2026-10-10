import subprocess
import sys

from voiceio.ibus import engine_pids


def test_engine_pids_match_exact_argv_not_lookalikes(tmp_path):
    log = tmp_path / "voiceio-ibus-engine.log"
    log.write_text("")
    sleeper = "import time; time.sleep(30)"
    # argv after -c is passed through untouched: [python, -c, code, -m, module]
    engine = subprocess.Popen([sys.executable, "-c", sleeper, "-m", "voiceio.ibus.engine"])
    lookalike = subprocess.Popen(["tail", "-f", str(log)])  # what pgrep -f used to kill
    try:
        pids = engine_pids()
        assert engine.pid in pids
        assert lookalike.pid not in pids
    finally:
        engine.kill()
        lookalike.kill()


def test_kill_stale_engines_waits_for_exit(tmp_path):
    from voiceio.ibus.session import _running, kill_stale_engines
    # Exits ~0.5s after SIGTERM — a slow engine whose cleanup used to run
    # after the successor had already bound the socket.
    code = ("import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, lambda *a: (time.sleep(0.5), sys.exit(0)))\n"
            "time.sleep(30)")
    slow = subprocess.Popen([sys.executable, "-c", code, "-m", "voiceio.ibus.engine"])
    import time
    time.sleep(0.3)  # let it install the handler
    kill_stale_engines(tmp_path / "sock")
    slow.wait(timeout=1)  # already gone (reaped here), not still exiting
    assert not _running(slow.pid)
