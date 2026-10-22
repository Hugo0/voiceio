#!/usr/bin/env bash
# Point the AUR package at a released version: pkgver, pkgrel, sha256sums and
# .SRCINFO. Run after the release is on PyPI (CI does, see publish.yml).
#
#   packaging/aur/update.sh 1.11.0 [PKGREL]
#
# Needs: curl, python3; makepkg if available (else .SRCINFO is patched).
set -euo pipefail

version=${1:?usage: $0 VERSION [PKGREL]}
pkgrel=${2:-1}
cd "$(dirname "$(realpath "$0")")"

wheel="python_voiceio-${version}-py3-none-any.whl"
# A fresh upload can take a minute to show up in the JSON API.
for _ in $(seq 30); do
    wheel_sha=$(curl -fsS "https://pypi.org/pypi/python-voiceio/${version}/json" 2>/dev/null |
        python3 -c "import json,sys; print(next(u['digests']['sha256'] for u in json.load(sys.stdin)['urls'] if u['filename']=='$wheel'))" 2>/dev/null) && break
    sleep 10
done
[[ -n ${wheel_sha:-} ]] || { echo "error: $wheel not on PyPI" >&2; exit 1; }
bundle_sha=$(sha256sum build-bundle.sh | cut -d' ' -f1)

python3 - "$version" "$pkgrel" "$wheel_sha" "$bundle_sha" <<'PY'
import re, sys
version, pkgrel, wheel_sha, bundle_sha = sys.argv[1:]
p = open("PKGBUILD").read()
p = re.sub(r"(?m)^pkgver=.*$", f"pkgver={version}", p)
p = re.sub(r"(?m)^pkgrel=.*$", f"pkgrel={pkgrel}", p)
p = re.sub(r"(?ms)^sha256sums=\(.*?\)",
           f"sha256sums=('{wheel_sha}'\n            '{bundle_sha}')", p)
open("PKGBUILD", "w").write(p)
PY

# makepkg is the reference, but Ubuntu's build dies on GitHub's runners; only
# the version, file names and checksums change between releases, so patch
# those when it can't run.
if ! { command -v makepkg >/dev/null && makepkg --printsrcinfo > .SRCINFO.new 2>/dev/null; }; then
    python3 - "$version" "$pkgrel" "$wheel_sha" "$bundle_sha" > .SRCINFO.new <<'PY'
import re, sys
version, pkgrel, wheel_sha, bundle_sha = sys.argv[1:]
s = open(".SRCINFO").read()
old = re.search(r"(?m)^\tpkgver = (.*)$", s)[1]
s = s.replace(f"python_voiceio-{old}-", f"python_voiceio-{version}-")
s = re.sub(r"(?m)^\tpkgver = .*$", f"\tpkgver = {version}", s)
s = re.sub(r"(?m)^\tpkgrel = .*$", f"\tpkgrel = {pkgrel}", s)
s = re.sub(r"(?m)(^\tsha256sums = .*\n)+",
           f"\tsha256sums = {wheel_sha}\n\tsha256sums = {bundle_sha}\n", s)
sys.stdout.write(s)
PY
fi
mv .SRCINFO.new .SRCINFO
echo "PKGBUILD and .SRCINFO now at ${version}-${pkgrel}"
