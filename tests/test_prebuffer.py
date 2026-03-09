"""Tests for the ring buffer and AudioRecorder pre-buffering logic."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from voiceio.config import AudioConfig
from voiceio.recorder import AudioRecorder, RingBuffer


class TestRingBuffer:
    def test_empty(self):
        rb = RingBuffer(100)
        result = rb.get()
        assert len(result) == 0

    def test_append_within_capacity(self):
        rb = RingBuffer(100)
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        rb.append(data)
        result = rb.get()
        np.testing.assert_array_equal(result, data)

    def test_append_wraps_around(self):
        rb = RingBuffer(4)
        rb.append(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        rb.append(np.array([4.0, 5.0, 6.0], dtype=np.float32))
        result = rb.get()
        # Should keep last 4: [3, 4, 5, 6]
        np.testing.assert_array_equal(result, [3.0, 4.0, 5.0, 6.0])

    def test_append_exact_capacity(self):
        rb = RingBuffer(3)
        rb.append(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        result = rb.get()
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_append_larger_than_capacity(self):
        rb = RingBuffer(3)
        rb.append(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
        result = rb.get()
        np.testing.assert_array_equal(result, [3.0, 4.0, 5.0])

    def test_clear(self):
        rb = RingBuffer(10)
        rb.append(np.array([1.0, 2.0], dtype=np.float32))
        rb.clear()
        assert len(rb.get()) == 0

    def test_fifo_order_preserved(self):
        rb = RingBuffer(5)
        for i in range(10):
            rb.append(np.array([float(i)], dtype=np.float32))
        result = rb.get()
        np.testing.assert_array_equal(result, [5.0, 6.0, 7.0, 8.0, 9.0])

    def test_multiple_small_appends(self):
        rb = RingBuffer(6)
        rb.append(np.array([1.0, 2.0], dtype=np.float32))
        rb.append(np.array([3.0, 4.0], dtype=np.float32))
        rb.append(np.array([5.0, 6.0], dtype=np.float32))
        result = rb.get()
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_prebuffer_duration(self):
        """1 second at 16kHz = 16000 samples."""
        sample_rate = 16000
        rb = RingBuffer(sample_rate)  # 1 second
        # Fill with 2 seconds of data
        for _ in range(20):
            rb.append(np.ones(1600, dtype=np.float32))
        result = rb.get()
        assert len(result) == sample_rate


class TestAudioRecorderPrebuffer:
    """Test that AudioRecorder correctly mixes 1D ring buffer output with 2D sounddevice chunks."""

    def test_prebuffer_concat_with_2d_chunks(self):
        """Ring buffer returns 1D, sounddevice gives 2D (frames, 1) - must not crash on concatenate."""
        cfg = AudioConfig(sample_rate=16000, prebuffer_secs=0.5)
        rec = AudioRecorder(cfg)

        # Simulate: fill ring buffer with some audio (via 2D input like sounddevice)
        for _ in range(10):
            fake_input = np.ones((1600, 1), dtype=np.float32)
            rec._ring.append(fake_input)

        # Start recording - grabs ring buffer into _chunks
        rec._recording = False
        rec._stream = MagicMock()  # pretend stream is open
        rec.start()

        # Simulate a sounddevice callback delivering 2D data
        fake_callback_data = np.ones((1024, 1), dtype=np.float32) * 0.5
        rec._callback(fake_callback_data, 1024, None, None)

        # Stop should concatenate without ValueError
        audio = rec.stop()
        assert audio is not None
        assert audio.ndim == 1
        # Should contain prebuffer + callback data
        assert len(audio) > 1024
