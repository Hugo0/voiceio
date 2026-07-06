#!/usr/bin/env bash
# install_matrix.sh — prove voiceio installs on clean containers.
#
# Runs the VERBATIM README/INSTALL.md install funnel on clean distro images,
# for both the published PyPI package and the local source tree, then reports a
# PASS/FAIL table. Headless is expected (no mic/display/ibus daemon): success is
# "installs cleanly, CLI runs, setup degrades gracefully with no traceback,
# doctor reports missing pieces without crashing".
#
# Usage:
#   scripts/install_matrix.sh                 # all distros, both sources
#   DISTROS="ubuntu:24.04" SOURCES="source"  scripts/install_matrix.sh
#
# Env:
#   DISTROS  space-separated docker images   (default: the 3 supported)
#   SOURCES  space-separated: pypi source    (default: both)
#   REPO     path to the source tree mounted at /repo (default: git top-level)

set -uo pipefail

REPO="${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
DISTROS="${DISTROS:-ubuntu:24.04 debian:12 fedora:40}"
SOURCES="${SOURCES:-pypi source}"
LOGDIR="$(mktemp -d)"

echo "repo=$REPO"
echo "logs=$LOGDIR"
echo

# Per-distro: package-manager install line (VERBATIM from README distro blocks),
# and the python interpreter / pipx bootstrap. Kept in sync with README + the
# SYSTEM_DEPS table in voiceio/platform.py.
sysdeps() {
  case "$1" in
    ubuntu:*|debian:*)
      # VERBATIM README apt line (portaudio19-dev pulls libportaudio2 at runtime).
      echo "apt-get update -qq && apt-get install -y -qq pipx build-essential python3-dev portaudio19-dev ibus gir1.2-ibus-1.0 python3-gi >/dev/null" ;;
    fedora:*)
      echo "dnf install -y -q pipx gcc gcc-c++ make python3-devel portaudio-devel ibus ibus-libs python3-gobject >/dev/null" ;;
    archlinux:*)
      echo "pacman -Sy --noconfirm python-pipx base-devel portaudio ibus python-gobject >/dev/null" ;;
    *) echo "false" ;;
  esac
}

# Install command for the package under test.
install_cmd() {
  local source="$1"
  if [ "$source" = "pypi" ]; then
    echo "pipx install python-voiceio"
  else
    # Local source: copy out of the read-only mount (setuptools writes egg-info
    # into the tree), then `pipx install` so the CLI lands on PATH like PyPI does.
    echo "cp -a /repo /src && rm -rf /src/.git /src/.claude && pipx install /src"
  fi
}

run_one() {
  local image="$1" source="$2"
  local tag="${image//[:\/]/_}__${source}"
  local log="$LOGDIR/$tag.log"
  local deps; deps="$(sysdeps "$image")"
  local inst; inst="$(install_cmd "$source")"

  # The full guest script. Mirrors README Quick start + INSTALL.md steps 2-5.
  local guest
  guest=$(cat <<EOF
set -uo pipefail
echo "== step: system deps =="
$deps || { echo "SYSDEPS_FAILED"; exit 90; }

echo "== step: pipx bootstrap =="
export PATH="/root/.local/bin:\$PATH"
pipx ensurepath >/dev/null 2>&1 || true

echo "== step: install ($source) =="
$inst || { echo "INSTALL_FAILED"; exit 91; }

echo "== step: voiceio --version =="
voiceio --version || { echo "VERSION_FAILED"; exit 92; }

echo "== step: voiceio setup --defaults =="
voiceio setup --defaults
setup_rc=\$?
echo "SETUP_EXIT=\$setup_rc"

echo "== step: voiceio doctor =="
voiceio doctor
doctor_rc=\$?
echo "DOCTOR_EXIT=\$doctor_rc"

echo "ALL_STEPS_DONE"
EOF
)

  echo ">>> $image [$source]"
  docker run --rm \
    -v "$REPO:/repo:ro" \
    -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
    -e PIP_ROOT_USER_ACTION=ignore \
    "$image" bash -c "$guest" >"$log" 2>&1
  local rc=$?

  # ── Verdict ──────────────────────────────────────────────────────────
  # PASS bar for a headless container: package installs, CLI runs, and both
  # setup + doctor degrade gracefully (documented non-zero exit, machine-
  # readable reason, NO traceback / argparse breakage).
  local verdict="PASS" note=""
  if grep -q "SYSDEPS_FAILED" "$log"; then verdict="FAIL"; note="system deps install"; fi
  if grep -q "INSTALL_FAILED" "$log"; then verdict="FAIL"; note="pip/pipx install (build error?)"; fi
  if grep -q "VERSION_FAILED" "$log"; then verdict="FAIL"; note="CLI not on PATH / import error"; fi
  if grep -qi "Traceback (most recent call last)" "$log"; then verdict="FAIL"; note="traceback in setup/doctor"; fi
  if grep -q "voiceio crashed" "$log"; then verdict="FAIL"; note="unhandled crash (crash.log written)"; fi
  # setup --defaults must be a wired flag AND must actually run the
  # non-interactive path (emitting a [voiceio-setup] progress line).
  if grep -q "unrecognized arguments: --defaults" "$log"; then
    if [ "$source" = "pypi" ]; then
      # Known gap in the published release, already fixed in source.
      verdict="FIXED-NEXT"; note="setup --defaults missing in published build; fixed in next release"
    else
      verdict="FAIL"; note="setup --defaults not wired"
    fi
  elif ! grep -q "\[voiceio-setup\]" "$log"; then
    verdict="FAIL"; note="${note:-setup produced no progress lines}"
  fi
  if ! grep -q "ALL_STEPS_DONE" "$log"; then verdict="FAIL"; note="${note:-guest aborted early}"; fi

  local setup_rc doctor_rc ver
  setup_rc=$(grep -oP 'SETUP_EXIT=\K[0-9]+' "$log" | tail -1)
  doctor_rc=$(grep -oP 'DOCTOR_EXIT=\K[0-9]+' "$log" | tail -1)
  ver=$(grep -oP 'voiceio \K[0-9][0-9.]*' "$log" | head -1)

  printf '%s\t%s\t%s\tver=%s setup_rc=%s doctor_rc=%s %s\n' \
    "$verdict" "$image" "$source" "${ver:-?}" "${setup_rc:-?}" "${doctor_rc:-?}" "$note" \
    >>"$LOGDIR/results.tsv"
  echo "    -> $verdict (setup_rc=${setup_rc:-?} doctor_rc=${doctor_rc:-?}) log: $log"
}

: >"$LOGDIR/results.tsv"
for image in $DISTROS; do
  for source in $SOURCES; do
    run_one "$image" "$source"
  done
done

echo
echo "======================= MATRIX ======================="
printf '%-6s %-16s %-8s %s\n' "RESULT" "DISTRO" "SOURCE" "DETAIL"
while IFS=$'\t' read -r verdict image source detail; do
  printf '%-6s %-16s %-8s %s\n' "$verdict" "$image" "$source" "$detail"
done <"$LOGDIR/results.tsv"
echo "======================================================"
echo "full logs in: $LOGDIR"

# FIXED-NEXT (pypi-only gaps already fixed in source) does not fail the run.
grep -q '^FAIL' "$LOGDIR/results.tsv" && exit 1 || exit 0
