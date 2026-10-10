#!/usr/bin/env bash
# Push the model you dictate with to another machine running voiceio (e.g. a
# home server serving your phone), and switch it over there. Meant as
# `[learn] on_promote`, which runs after every promote/rollback with the model
# in $VOICEIO_MODEL:
#
#   [learn]
#   on_promote = "~/voiceio/scripts/push_model.sh myserver --reload-unit my-voice-reload.service"
#
# On HOST the model lands under ~/.local/share/voiceio/learn/models/<run>/ct2,
# and two relative links flip atomically: `current` (point the server at it,
# e.g. VOICE_MODEL=%h/.local/share/voiceio/learn/models/current) and
# `previous`. The copy completes before the flip, so the server never sees a
# half-copied model. Then --reload-unit, if given, is started on HOST to switch
# the server over (when idle, rolling back if it won't load — that's the
# server's job). Nothing is ever deleted there.
set -euo pipefail
host="${1:?usage: $0 HOST [--reload-unit UNIT]  (model from \$VOICEIO_MODEL)}"
shift
unit=""
if [ "${1:-}" = "--reload-unit" ]; then unit="${2:?--reload-unit needs a unit name}"; fi
model="${VOICEIO_MODEL:-$(voiceio config get model.name)}"

if [ ! -f "$model/model.bin" ]; then
  echo "push_model: '$model' is not a local fine-tune; nothing to push."
  exit 0
fi
run="$(basename "$(dirname "$model")")"
dest=".local/share/voiceio/learn/models/$run/ct2"

ssh "$host" "umask 077; mkdir -p '$dest'"
rsync -a --partial --chmod=D700,F600 "$model/" "$host:$dest/"
ssh "$host" "set -e; cd .local/share/voiceio/learn/models
  [ -L current ] && ln -sfn \"\$(readlink current)\" previous.tmp && mv -T previous.tmp previous
  ln -sfn '$run/ct2' current.tmp && mv -T current.tmp current"
echo "pushed $run → $host (current)"
if [ -n "$unit" ]; then
  ssh "$host" "systemctl --user start --no-block '$unit'"
  echo "started $unit on $host"
fi
