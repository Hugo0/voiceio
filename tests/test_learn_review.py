

def test_queue_all_covers_every_segment_of_unreviewed_clips():
    from voiceio.learn import review
    from voiceio.learn.store import Clip, Label, Segment
    a = Clip("a.wav", 0, 4, "", "")
    b = Clip("b.wav", 0, 4, "", "")
    segs = (Segment(0, 2, "one"), Segment(2, 4, "two"))
    labels = {"a.wav": Label("a.wav", segs, "teacher:x", 0),
              "b.wav": Label("b.wav", segs, "human", 0)}
    items = review.queue_all([a, b], labels)
    assert [(i.clip.audio, i.index) for i in items] == [("a.wav", 0), ("a.wav", 1)]


def test_live_span_shows_the_live_decode_only_where_it_differs():
    from voiceio.learn.review import live_span
    live = "okay so we use kalchi daily and then more stuff"
    assert live_span("We use Kalshi daily.", live) == "we use kalchi daily"
    assert live_span("We use Kalshi daily.", "so we use Kalshi daily ok") is None
    assert live_span("totally different words here", "nothing alike at all") is None
