#!/usr/bin/env bash
# Bring dictation recorded by your voiceio server (phone → `python -m
# voiceio.server`) home for `voiceio learn`, over SSH. Nothing else is involved.
#
#   scripts/pull_phone_dictation.sh myserver
#
# The server only keeps audio if its owner opted in there, with
#   [data]
#   retain_audio = true
# in the server's ~/.config/voiceio/config.toml. Only import your own voice:
# the model learns whoever is speaking. Safe to re-run — imports are
# content-addressed, so clips already imported are skipped.
set -euo pipefail
host="${1:?usage: $0 HOST}"
dest="${XDG_DATA_HOME:-$HOME/.local/share}/voiceio/learn/incoming/$host"
mkdir -p -m 700 "$dest"
remote='${XDG_STATE_HOME:-$HOME/.local/state}/voiceio/recordings/'
rsync -a --chmod=D700,F600 "$host:$remote" "$dest/"
voiceio learn import "$dest" --source phone
