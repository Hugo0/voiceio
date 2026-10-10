

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
