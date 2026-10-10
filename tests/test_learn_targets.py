"""Training targets must match what Whisper emits at inference."""
import pytest

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.WhisperTokenizer.from_pretrained(
            "openai/whisper-small", language="en", task="transcribe", local_files_only=True)
    except OSError:
        pytest.skip("whisper-small tokenizer not cached")


def test_text_target_has_leading_space(tok):
    from voiceio.learn.train import target_ids
    ids = target_ids(tok, {"text": "hello world"}, timestamps=False)
    toks = tok.convert_ids_to_tokens(ids)
    assert toks[:4] == ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
    assert toks[4] == "Ġhello" and toks[-1] == "<|endoftext|>"


def test_timestamp_target_interleaves_parts(tok):
    from voiceio.learn.train import target_ids
    row = {"text": "hi there. bye", "parts": [[0.0, 1.0, "hi there."], [1.5, 2.04, "bye"]]}
    ids = target_ids(tok, row, timestamps=True)
    assert tok.decode(ids, decode_with_timestamps=True) == (
        "<|startoftranscript|><|en|><|transcribe|><|0.00|> hi there.<|1.00|>"
        "<|1.50|> bye<|2.04|><|endoftext|>")
    # Synthetic rows have no parts: plain text target even when timestamping.
    assert "<|notimestamps|>" in tok.convert_ids_to_tokens(
        target_ids(tok, {"text": "bye"}, timestamps=True))
