#!/usr/bin/env bash
# Push PKGBUILD, .SRCINFO and build-bundle.sh to the AUR (python-voiceio).
#
#   packaging/aur/publish.sh
#
# Run update.sh first. Needs an AUR account whose SSH key is loaded (or
# GIT_SSH_COMMAND set, as CI does). The first push creates the package.
set -euo pipefail
here=$(dirname "$(realpath "$0")")
repo=${AUR_REPO:-ssh://aur@aur.archlinux.org/python-voiceio.git}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git clone -q "$repo" "$work/aur"
# -L: build-bundle.sh is a symlink here; the AUR needs the file.
cp -L "$here/PKGBUILD" "$here/.SRCINFO" "$here/build-bundle.sh" "$work/aur/"
cd "$work/aur"
git add PKGBUILD .SRCINFO build-bundle.sh
if git diff --cached --quiet; then
    echo "AUR already up to date"
    exit 0
fi
ver=$(sed -n 's/^\tpkgver = //p' .SRCINFO)-$(sed -n 's/^\tpkgrel = //p' .SRCINFO)
git commit -q -m "Update to $ver"
git push -q origin HEAD:master
echo "pushed python-voiceio $ver to the AUR"
