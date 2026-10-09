#!/usr/bin/env bash
# Copy your vocabulary and corrections to another machine running voiceio
# (e.g. a home server serving `python -m voiceio.server` to your phone), so
# remote dictation gets the same names and fixes as the desktop.
#
#   scripts/sync_personalization.sh myserver            # one-way, this → there
#
# Only the two hand-curated files are copied. Usage statistics
# (vocab_stats.json) stay per machine; nothing else is sent.
set -euo pipefail
host="${1:?usage: $0 HOST}"
src="${XDG_CONFIG_HOME:-$HOME/.config}/voiceio"
ssh "$host" 'umask 077; mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}/voiceio"'
for f in vocabulary.txt corrections.json; do
  [ -f "$src/$f" ] || continue
  scp -q -p "$src/$f" "$host:.config/voiceio/$f"
  echo "synced $f"
done
echo "The server reloads both on the next request (mtime-cached); no restart needed."
