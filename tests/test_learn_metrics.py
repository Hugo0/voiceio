from voiceio.learn.metrics import align, count_repetitions, term_hits, wer


def test_align_counts_error_types():
    a = align("alpha bravo charlie delta", "alpha xray charlie delta echo")
    assert (a.substitutions, a.deletions, a.insertions) == (1, 0, 1)
    d = align("alpha bravo charlie", "alpha charlie")
    assert (d.substitutions, d.deletions, d.insertions) == (0, 1, 0)
    assert abs(align("the cat sat on the mat", "the cat sit on mat today").wer - 3 / 6) < 1e-9


def test_wer_ignores_case_and_punctuation():
    assert wer("Hello, world.", "hello world") == 0.0


def test_wer_empty_reference():
    assert wer("", "") == 0.0
    assert wer("", "noise") == 1.0


def test_term_hits_multiword_and_capped():
    ref = "we use Kalshi and HyperNEAT, Kalshi again"
    hyp = "we use kalchi and HyperNEAT HyperNEAT HyperNEAT"
    assert term_hits(ref, hyp, ["Kalshi", "HyperNEAT", "Hetzner"]) == (3, 1)
    assert term_hits("open the Claude Code app", "open the claude code app",
                     ["Claude Code"]) == (1, 1)


def test_repetition_detects_long_period_loop():
    loop = "because it should not work out in the complex world " * 4
    assert count_repetitions(loop) == 1
    assert count_repetitions("no no that is fine") == 0
