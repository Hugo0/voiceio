#!/usr/bin/env bash
# Build voiceio_<version>-1_amd64.deb from a voiceio wheel.
#
#   packaging/deb/build-deb.sh dist/python_voiceio-X.Y.Z-py3-none-any.whl OUTDIR
#
# One package for every Debian/Ubuntu release: it bundles its own Python (see
# ../build-bundle.sh), so it does not depend on the distro's python3 version.
# Build it on the oldest supported release (CI uses ubuntu:22.04) so the one
# compiled extension (evdev) links against the oldest glibc.
#
# Needs: uv, gcc, linux-libc-dev, dpkg-deb, libportaudio2 (for the smoke test).
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 WHEEL OUTDIR" >&2
    exit 2
fi
wheel=$(realpath "$1")
outdir=$(realpath -m "$2")
here=$(dirname "$(realpath "$0")")

version=$(basename "$wheel" | cut -d- -f2)
revision=${DEB_REVISION:-1}
arch=$(dpkg --print-architecture)

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
root="$work/root"

"$here/../build-bundle.sh" "$root" "$wheel" deb

doc="$root/usr/share/doc/voiceio"
mkdir -p "$doc"
cp "$here/copyright" "$doc/copyright"

mkdir -p "$root/DEBIAN"
installed_kb=$(du -sk --exclude=DEBIAN "$root" | cut -f1)
sed -e "s/@VERSION@/$version-$revision/" \
    -e "s/@ARCH@/$arch/" \
    -e "s/@INSTALLED_SIZE@/$installed_kb/" \
    "$here/control.in" > "$root/DEBIAN/control"

chmod -R u+rwX,go+rX,go-w "$root"

mkdir -p "$outdir"
deb="$outdir/voiceio_${version}-${revision}_${arch}.deb"
dpkg-deb --root-owner-group -Zxz --build "$root" "$deb"
dpkg-deb --info "$deb" | sed -n '/Package:/,$p'
echo "built $deb"
