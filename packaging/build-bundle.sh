#!/usr/bin/env bash
# Build a self-contained voiceio install tree for a distro package.
#
#   packaging/build-bundle.sh DESTDIR WHEEL PACKAGER
#
# Lays out DESTDIR/usr/lib/voiceio: a standalone CPython (python-build-standalone,
# fetched by uv) with voiceio[desktop,tts] and all its Python dependencies
# installed into it, plus /usr/bin/voiceio and /usr/bin/voiceio-toggle symlinks.
#
# Why bundle the interpreter instead of using the distro's python3: the heavy
# dependencies (ctranslate2, onnxruntime, PyAV, tokenizers) are binary wheels
# for one CPython ABI. A venv on top of /usr/bin/python3 breaks the day the
# distro bumps python3 (Arch does it yearly; Ubuntu with every release), and a
# .deb built against one Ubuntu's python3 would not install on the next. The
# bundled interpreter makes one package work across distro releases and never
# break on a python3 upgrade. The distro's python3 is still used, but only for
# the two small processes that need its GObject bindings (IBus engine, tray);
# both import only the standard library, gi and voiceio's pure-Python code.
#
# PACKAGER (deb, aur, ...) is written to usr/lib/voiceio/voiceio-packaged so
# voiceio knows the package manager owns it (no ~/.local/bin symlinks, no pip
# installs into it, and "upgrade/uninstall with your package manager" hints).
#
# Needs: uv, a C compiler and kernel headers (evdev has no binary wheel).
# Env: VOICEIO_PYTHON (default 3.12), VOICEIO_EXTRAS (default desktop,tts).
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 DESTDIR WHEEL PACKAGER" >&2
    exit 2
fi
dest=$(realpath -m "$1")
wheel=$(realpath "$2")
packager=$3

prefix=/usr/lib/voiceio
pyver=${VOICEIO_PYTHON:-3.12}
extras=${VOICEIO_EXTRAS:-desktop,tts}
root="$dest$prefix"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
# Keep uv's cache and downloads out of the builder's home directory.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$work/cache}"
export UV_PYTHON_PREFERENCE=only-managed UV_NO_CONFIG=1

rm -rf "$root"
mkdir -p "$root"
uv python install --no-bin --install-dir "$work/python" "$pyver"
src=$(find "$work/python" -mindepth 1 -maxdepth 1 -type d -name 'cpython-*' | head -n1)
cp -a "$src/." "$root/"
py="$root/bin/python3"

# Unused by voiceio; a third of the interpreter's size.
rm -rf "$root"/lib/python3.*/{test,idlelib,tkinter,turtledemo,ensurepip,lib2to3} \
       "$root"/lib/{libtcl,libtk,tcl,tk,itcl,thread}* "$root/share"
rm -f "$root"/lib/python3.*/EXTERNALLY-MANAGED

uv pip install --python "$py" --link-mode copy \
    "$wheel[$extras]"
# Headers were only needed to compile evdev; direct_url.json records the
# wheel's build-machine path.
rm -rf "$root/include"
rm -f "$root"/lib/python3.*/site-packages/*.dist-info/direct_url.json

# Extensions compiled here (evdev) carry the build path in their debug info.
{ grep -rlF --null --include='*.so' -e "$UV_CACHE_DIR" "$root" || true; } |
    xargs -0 -r strip --strip-debug
# Entry-point shebangs (and any other text file) carry the build path.
{ grep -rlI --null -F "$dest" "$root" || true; } | xargs -0 -r sed -i "s|$dest||g"

# Byte-compile with the final paths so tracebacks and the pyc files are right
# and nothing tries to write __pycache__ into a root-owned tree at run time.
"$py" -m compileall -q -f -j0 -s "$dest" -p / "$root/lib" >/dev/null

printf '%s\n' "$packager" > "$root/voiceio-packaged"

mkdir -p "$dest/usr/bin"
for name in voiceio voiceio-toggle; do
    ln -sf "../lib/voiceio/bin/$name" "$dest/usr/bin/$name"
done
# Root-owned in the package: never group/world-writable, whatever the umask.
chmod -R u+rwX,go+rX,go-w "$dest/usr"

# Fail the build rather than ship a tree that points into the build directory.
if leaked=$(grep -rlF -e "$dest" -e "$(dirname "$wheel")" -e "$UV_CACHE_DIR" "$root"); then
    echo "error: build path left in:" >&2
    echo "$leaked" >&2
    exit 1
fi
"$py" -c "import voiceio, faster_whisper, sounddevice, evdev; print('voiceio', voiceio.__version__, 'bundle ok')"
