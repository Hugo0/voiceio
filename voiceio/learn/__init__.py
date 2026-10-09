"""Personalization: learn the user's words from the user's own dictation.

Everything here is local and opt-in. The loop:

    capture  — keep (audio, raw decoder text, final text) per utterance
    label    — a stronger teacher model transcribes the clips once
    review   — the user fixes only the spans that matter (vocabulary, disagreements)
    augment  — synthesize extra examples of hard terms with TTS
    train    — LoRA-fine-tune the decoder (optional `[train]` extra / rented GPU)
    eval     — held-out personal clips + a public slice; promote only on a win
    maintain — prune vocabulary and retire dead correction rules by real usage

No user data ships with voiceio; every path resolves under the user's XDG dirs.
"""
