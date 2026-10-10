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
