"""Tests for the IBus engine's pre-ready command buffer (fix #7).

The buffer must never replay commands that went stale before the engine
instance existed, and must be cleared outright when a fresh engine is created.
"""
from __future__ import annotations

from voiceio.ibus.pending import PendingBuffer


def test_fresh_commands_are_replayed_in_order():
    buf = PendingBuffer(max_age=3.0)
    buf.add("preedit:a", now=100.0)
    buf.add("commit:b", now=100.5)
    assert buf.drain_fresh(now=101.0) == ["preedit:a", "commit:b"]
    assert len(buf) == 0  # draining empties the buffer


def test_stale_commands_are_dropped():
    buf = PendingBuffer(max_age=3.0)
    buf.add("commit:old", now=100.0)   # 5s before drain -> stale
    buf.add("commit:new", now=104.0)   # 1s before drain -> fresh
    assert buf.drain_fresh(now=105.0) == ["commit:new"]


def test_all_stale_yields_nothing():
    buf = PendingBuffer(max_age=3.0)
    buf.add("commit:x", now=0.0)
    assert buf.drain_fresh(now=100.0) == []


def test_clear_empties_buffer():
    buf = PendingBuffer()
    buf.add("preedit:z")
    assert len(buf) == 1
    buf.clear()
    assert len(buf) == 0
    assert buf.drain_fresh() == []
