"""On-screen vocabulary: term extraction, budget share, capture deadline, and
that capture never sits on the hotkey path."""
from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from voiceio import context_terms
from voiceio.context_terms import ScreenContext, capture, extract_terms
from voiceio.core.bias import SCREEN_TOKEN_SHARE, BiasProvider
from voiceio.core.decoders.base import BiasMode, Capabilities
from voiceio.tokens import count_tokens

HOTWORDS = Capabilities(bias=BiasMode.HOTWORDS | BiasMode.PROMPT, max_bias_tokens=120)


# ── extraction ────────────────────────────────────────────────────────────


class TestExtract:
    def test_rare_names_camelcase_acronyms(self):
        terms = extract_terms(["Kalshi pricing - Discord",
                               "The HyperNEAT paper uses OpenRouter and the OKR doc."])
        assert {"Kalshi", "HyperNEAT", "OpenRouter", "OKR"} <= set(terms)

    def test_common_words_and_app_names_are_not_terms(self):
        terms = extract_terms(["Inbox - Google Chrome",
                               "Please check the API before the meeting with Hugo."])
        assert terms == []

    def test_multiword_name_but_not_a_greeting(self):
        terms = extract_terms(["", "Hey Kosh, ask Tarrant Fnordwick about it."])
        assert "Tarrant Fnordwick" in terms
        assert "Kosh" in terms
        assert not any(t.startswith("Hey") for t in terms)
        # A name and its own parts would spend tokens twice on one word.
        assert "Tarrant" not in terms and "Fnordwick" not in terms

    def test_identifiers_split_numbers_dropped(self):
        terms = extract_terms(["", "see github.com/Hugo0/voiceio and v2 of x86_64 build 2026"])
        assert "voiceio" in terms
        assert not any(any(c.isdigit() for c in t) for t in terms)

    def test_exclude_is_case_insensitive_and_dedupes_spellings(self):
        terms = extract_terms(["", "Kalshi kalshi KALSHI Zorblax"], exclude=["zorblax"])
        assert [t.lower() for t in terms] == ["kalshi"]

    def test_vocabulary_long_tail_ranks_first(self):
        terms = extract_terms(["", "Zorblax met Quuxley at Wibblecorp."],
                              prefer=["Wibblecorp"])
        assert terms[0] == "Wibblecorp"

    def test_title_beats_selection_on_ties(self):
        terms = extract_terms(["Quuxley", "Zorblax"])
        assert terms == ["Quuxley", "Zorblax"]

    def test_repeats_outrank_one_offs(self):
        terms = extract_terms(["", "Zorblax once. Quuxley, Quuxley, Quuxley."])
        assert terms[0] == "Quuxley"

    def test_capped(self):
        text = " ".join(f"Zorb{chr(97 + i)}lax" for i in range(20))
        assert len(extract_terms(["", text], max_terms=5)) == 5

    def test_huge_selection_is_cheap(self):
        text = "the quick brown fox Zorblax jumps " * 20000
        t0 = time.monotonic()
        assert extract_terms(["", text]) == ["Zorblax"]
        assert time.monotonic() - t0 < 1.0

    def test_empty(self):
        assert extract_terms(["", ""]) == []


# ── budget share in the hotwords ──────────────────────────────────────────


def _provider(terms: list[str]) -> BiasProvider:
    from voiceio import config
    from voiceio.vocabulary import VocabularyLoader
    config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (config.CONFIG_DIR / "vocabulary.txt").write_text("\n".join(terms) + "\n")
    return BiasProvider(VocabularyLoader(config.ModelConfig()))


def _vocab(n: int) -> list[str]:
    # Distinct pronounceable made-up names, in file (= rank) order.
    syll = ["ka", "zor", "bli", "mun", "tep", "vas", "quo", "rin", "lud", "fey"]
    return [(syll[i % 10] + syll[(i // 10) % 10] + syll[(i * 7) % 10]).capitalize() + "x"
            for i in range(n)]


class TestBudget:
    def test_no_screen_terms_is_unchanged(self):
        bp = _provider(_vocab(60))
        assert bp.bias_for(HOTWORDS).hotwords == bp.bias_for(HOTWORDS, []).hotwords

    def test_screen_terms_displace_only_the_vocabulary_tail(self):
        vocab = _vocab(60)
        bp = _provider(vocab)
        base = bp.bias_for(HOTWORDS).hotwords.split(", ")
        merged = bp.bias_for(HOTWORDS, ["Quuxley", "Wibblecorp"]).hotwords.split(", ")
        assert "Quuxley" in merged and "Wibblecorp" in merged
        kept = [t for t in merged if t in vocab]
        assert kept == base[:len(kept)], "the best-ranked terms survive, in order"
        assert len(kept) < len(base)
        assert count_tokens(", ".join(merged), "small") <= 120 + 2  # join slack

    def test_screen_share_is_capped(self):
        bp = _provider(_vocab(60))
        many = [f"Zorb{chr(97 + i)}laxian" for i in range(20)]
        merged = bp.bias_for(HOTWORDS, many).hotwords.split(", ")
        screen = [t for t in merged if t in many]
        assert 0 < len(screen) < len(many)
        assert count_tokens(", ".join(screen), "small") <= SCREEN_TOKEN_SHARE
        # The vocabulary keeps the rest of the budget.
        assert len(merged) - len(screen) >= 10

    def test_screen_terms_skip_what_hotwords_carry(self):
        bp = _provider(["Kalshi", "HyperNEAT"])
        assert bp.screen_terms(["Kalshi - Discord", "Quuxley and HyperNEAT"], HOTWORDS) == ["Quuxley"]

    def test_unbounded_appends(self):
        bp = _provider(_vocab(60))
        caps = Capabilities(bias=BiasMode.UNBOUNDED)
        merged = bp.bias_for(caps, ["Quuxley"]).hotwords.split(", ")
        assert merged[-1] == "Quuxley" and len(merged) == 61

    def test_decoder_without_hotwords_gets_nothing(self):
        bp = _provider(["Kalshi"])
        assert bp.screen_terms(["Quuxley"], Capabilities(bias=BiasMode.PROMPT)) == []

    def test_extraction_failure_degrades_to_nothing(self):
        bp = _provider(["Kalshi"])
        with patch("voiceio.context_terms.extract_terms", side_effect=RuntimeError):
            assert bp.screen_terms(["Quuxley"], HOTWORDS) == []


# ── capture: deadline and degrade ─────────────────────────────────────────


class TestCapture:
    def test_slow_source_is_dropped_at_the_deadline(self):
        gate = threading.Event()

        def slow(_t):
            gate.wait(5)
            return "Zorblax"

        t0 = time.monotonic()
        ctx = capture(0.1, readers={"title": lambda _t: "Kalshi", "selection": slow})
        elapsed = time.monotonic() - t0
        gate.set()
        assert elapsed < 0.5
        assert ctx.title == "Kalshi" and ctx.selection == ""

    def test_failing_source_degrades_silently(self):
        def boom(_t):
            raise OSError("no display")
        ctx = capture(0.1, readers={"title": boom, "selection": lambda _t: "x"})
        assert ctx == ScreenContext(title="", selection="x", secs=ctx.secs)

    def test_no_tools_no_selection(self, monkeypatch):
        monkeypatch.setattr(context_terms.shutil, "which", lambda _n: None)
        assert context_terms._read_primary(0.1) == ""

    def test_gnome_wayland_never_uses_wl_paste(self, monkeypatch):
        """wl-paste can take focus on GNOME and drop the IBus preedit."""
        from types import SimpleNamespace
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setattr("voiceio.platform.detect",
                            lambda: SimpleNamespace(is_wayland=True, desktop="gnome"))
        monkeypatch.setattr(context_terms.shutil, "which", lambda n: f"/usr/bin/{n}")
        with patch.object(context_terms.subprocess, "run") as run:
            assert context_terms._read_primary(0.1) == ""
        run.assert_not_called()

    def test_selection_read_has_the_deadline_as_timeout(self, monkeypatch):
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setattr(context_terms.shutil, "which", lambda n: f"/usr/bin/{n}")
        with patch.object(context_terms.subprocess, "run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "Zorblax".encode()
            assert context_terms._read_primary(0.15) == "Zorblax"
        assert run.call_args.kwargs["timeout"] == 0.15
        assert run.call_args.args[0][:2] == ["xclip", "-o"]


# ── the daemon: never on the hotkey path ──────────────────────────────────


@pytest.fixture
def vio():
    from test_app_wiring import _make_vio
    from fakes import FakeDecoder
    dec = FakeDecoder()
    dec.transcribe.return_value = "hello"
    v, _, _ = _make_vio(dec)
    v.cfg.model.screen_vocabulary = True
    yield v, dec
    if v._session is not None:
        v._session.cancel()


class TestDaemon:
    def test_capture_does_not_block_recording_start(self, vio):
        v, dec = vio
        gate = threading.Event()
        captured = threading.Event()

        def blocked_capture(*_a, **_k):
            gate.wait(5)
            captured.set()
            return ScreenContext(title="Quuxley - Discord")

        with patch("voiceio.context_terms.capture", side_effect=blocked_capture):
            t0 = time.monotonic()
            v.on_hotkey()
            assert time.monotonic() - t0 < 1.0
            assert v.recorder.is_recording, "recording started while capture blocks"
            assert "Quuxley" not in (v._recording_bias.hotwords or "")
            gate.set()
            assert captured.wait(5)
            for _ in range(100):
                if "Quuxley" in (v._recording_bias.hotwords or ""):
                    break
                time.sleep(0.02)
        assert "Quuxley" in (v._recording_bias.hotwords or "")
        assert "Quuxley" in (v._session._bias.hotwords or ""), "applied to later passes"

    def test_late_capture_never_touches_the_next_recording(self, vio):
        v, _ = vio
        gate = threading.Event()
        with patch("voiceio.context_terms.capture",
                   side_effect=lambda *a, **k: (gate.wait(5), ScreenContext(title="Quuxley"))[1]):
            v.on_hotkey()
            rid = v._recording
            v._recording += 1   # as if this recording ended and another began
            gate.set()
            time.sleep(0.3)
        v._recording = rid
        assert "Quuxley" not in (v._recording_bias.hotwords or "")

    def test_off_reads_nothing(self, vio):
        v, _ = vio
        v.cfg.model.screen_vocabulary = False
        with patch("voiceio.context_terms.capture") as cap:
            v.on_hotkey()
            time.sleep(0.1)
        cap.assert_not_called()


def _fake_tmux(outputs):
    calls = []

    def run(args, timeout):
        calls.append(args)
        return outputs.get(args[0], "")
    return run, calls


def test_tmux_session_names_become_words(monkeypatch):
    """Session names are the terminal's tabs: their slug parts are terms."""
    from voiceio import context_terms as c
    run, calls = _fake_tmux({"list-sessions": "s claude-swarmmemo-live\ns claude-wordle-native\nc %1\nc %2\n"})
    monkeypatch.setattr(c, "_tmux", run)
    monkeypatch.setattr(c.shutil, "which", lambda name: "/usr/bin/tmux")
    text = c._read_tmux(0.2)
    assert "swarmmemo" in text.split() and "claude-swarmmemo-live" not in text
    # Two clients: tmux can't say which one has focus, so no pane is read.
    assert not any(a[0] == "capture-pane" for a in calls)
    assert "swarmmemo" in c.extract_terms([text])


def test_tmux_pane_read_only_with_a_single_client(monkeypatch):
    from voiceio import context_terms as c
    run, calls = _fake_tmux({"list-sessions": "s work\nc %3\n", "capture-pane": "run xdotool now\n"})
    monkeypatch.setattr(c, "_tmux", run)
    monkeypatch.setattr(c.shutil, "which", lambda name: "/usr/bin/tmux")
    text = c._read_tmux(0.2)
    assert ["capture-pane", "-p", "-J", "-t", "%3"] in calls
    assert "xdotool" in text


def test_no_tmux_no_terminal_text(monkeypatch):
    from voiceio import context_terms as c
    monkeypatch.setattr(c.shutil, "which", lambda name: None)
    assert c._read_tmux(0.2) == ""
