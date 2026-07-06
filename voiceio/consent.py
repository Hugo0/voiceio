"""Explicit consent for cloud LLM calls.

voiceio is local-first: transcription never leaves the machine. Two optional
features (final-transcript polish and the weekly correction-mining review) can
send *text* (never audio) to a cloud LLM the user configures. Those calls are
gated on a recorded, explicit consent so an API key sitting in the environment
is never silently adopted.

Consent is a tiny JSON file under CONFIG_DIR. The wizard's consent UI and the
`voiceio correct` interactive flow call ``record_consent()``; every cloud call
checks ``has_cloud_consent()`` and fails open to local-only behaviour when it
is absent.
"""
from __future__ import annotations

import json
import logging
import time

from voiceio import config

log = logging.getLogger(__name__)


def has_cloud_consent() -> bool:
    """True if the user has explicitly allowed cloud LLM calls."""
    try:
        data = json.loads(config.CONSENT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("cloud"))


def record_consent(*, source: str = "user") -> None:
    """Persist explicit cloud consent. Idempotent; safe to call repeatedly.

    `source` is a free-form note ("wizard", "correct", "configured-key") kept
    for the user's own audit of how consent was granted.
    """
    if has_cloud_consent():
        return
    payload = {"cloud": True, "granted_at": time.time(), "source": source}
    try:
        config.secure_write(
            config.CONSENT_PATH, json.dumps(payload, indent=2) + "\n",
        )
        log.info("Recorded cloud consent (source=%s)", source)
    except OSError as e:
        log.warning("Could not record consent: %s", e)


def revoke_consent() -> None:
    """Remove any recorded cloud consent."""
    try:
        config.CONSENT_PATH.unlink(missing_ok=True)
    except OSError:
        pass
