#!/usr/bin/env bash
# Copy your vocabulary and corrections to another machine running voiceio
# (e.g. a home server serving `python -m voiceio.server` to your phone), so
# remote dictation gets the same names and fixes as the desktop.
#
#   scripts/sync_personalization.sh myserver            # one-way, this → there
#   scripts/sync_personalization.sh myserver --model    # also your fine-tuned model
#
# Only the two hand-curated files are copied, plus — with --model — the
# promoted fine-tune's weights (`voiceio learn promote`): never recordings,
# labels or history. Usage statistics (vocab_stats.json) stay per machine.
set -euo pipefail
host="${1:?usage: $0 HOST [--model]}"
with_model="${2:-}"
src="${XDG_CONFIG_HOME:-$HOME/.config}/voiceio"
ssh "$host" 'umask 077; mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}/voiceio"'
for f in vocabulary.txt corrections.json; do
  [ -f "$src/$f" ] || continue
  scp -q -p "$src/$f" "$host:.config/voiceio/$f"
  echo "synced $f"
done
echo "The server reloads both on the next request (mtime-cached); no restart needed."

[ "$with_model" = "--model" ] || exit 0
model="$(voiceio config get model.name 2>/dev/null || true)"
if [ ! -f "$model/model.bin" ]; then
  echo "--model: the active model ('${model:-?}') is not a local fine-tune; nothing to copy."
  exit 0
fi
case "$model" in
  "$HOME"/*) rel="${model#"$HOME"/}" ;;
  *) echo "--model: $model is outside \$HOME; copy it by hand." >&2; exit 1 ;;
esac
ssh "$host" "umask 077; mkdir -p '$rel'"
rsync -a --chmod=D700,F600 "$model/" "$host:$rel/"
echo "synced model → $host:~/$rel"
echo "Point the server at it — VOICE_MODEL=\$HOME/$rel (or [model] name in its config.toml) — and restart it."
