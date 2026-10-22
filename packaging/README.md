# Distro packages

Two packages, one build: `build-bundle.sh` lays out `/usr/lib/voiceio` (a
standalone CPython from python-build-standalone, fetched by uv, with
`python-voiceio[desktop,tts]` and every dependency installed into it) plus
`/usr/bin/voiceio` and `/usr/bin/voiceio-toggle`. The `.deb` and the AUR
package only wrap that tree.

**Why bundle Python.** faster-whisper, ctranslate2, tokenizers and wordfreq
aren't in the Arch repos (the AUR has ctranslate2 and tokenizers as
from-source C++/Rust builds; wordfreq isn't there at all), and Debian has none
of them. A venv on the distro's `python3` would work until the distro bumps
`python3`, then die on import (binary wheels are per-ABI); a `.deb` built for
one Ubuntu's `python3` wouldn't install on the next. The bundled interpreter
costs ~30 MB and makes one `.deb` work on every Debian/Ubuntu release and the
Arch package survive Python upgrades. The distro's `python3` + PyGObject is
still used for the IBus engine and the tray, which import only the stdlib, `gi`
and voiceio's own pure-Python modules.

`usr/lib/voiceio/voiceio-packaged` (contents: `deb` or `aur`) tells voiceio the
package manager owns the install: `voiceio setup` makes no `~/.local/bin`
symlinks and never pip-installs into it, and `voiceio uninstall` names the
package-manager command. The systemd user unit `voiceio setup` writes runs
`/usr/lib/voiceio/bin/voiceio`, which is the same path after every upgrade.

Size: ~210 MB `.deb`, ~260 MB `.pkg.tar.zst`, ~700 MB installed (the same as a
pipx install; av, ctranslate2, onnxruntime and wordfreq are most of it).

## .deb

Built by the `deb` job in `.github/workflows/publish.yml` on every `v*` tag, in
an `ubuntu:22.04` container (so it needs glibc >= 2.35), and attached to the
GitHub Release with its SHA-256 in the notes. Nothing to do by hand.

Locally (needs uv, gcc, dpkg-deb, libportaudio2):

```bash
uv build --wheel --out-dir dist
packaging/deb/build-deb.sh dist/python_voiceio-*.whl deb
sudo apt install ./deb/voiceio_*_amd64.deb
```

## AUR (`python-voiceio`)

`aur/PKGBUILD` downloads the release wheel from PyPI and runs
`build-bundle.sh` (shipped next to it in the AUR repo; `aur/build-bundle.sh` is
a symlink here). On every tag the `aur` job runs `aur/update.sh X.Y.Z` (new
`pkgver`, `pkgrel=1`, `sha256sums`, `.SRCINFO`), attaches `PKGBUILD` and
`SRCINFO` to the release, and pushes to the AUR if the `AUR_SSH_PRIVATE_KEY`
secret is set.

### One-time setup (Hugo)

1. Create an account at https://aur.archlinux.org/register and add an SSH
   public key under *My Account*. A dedicated key is best, since CI will hold it:

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/aur -C "voiceio AUR" -N ""
   cat ~/.ssh/aur.pub          # paste into the AUR account page
   ```

2. Tell ssh to use it for the AUR:

   ```bash
   cat >> ~/.ssh/config <<'EOF'
   Host aur.archlinux.org
     IdentityFile ~/.ssh/aur
     User aur
   EOF
   ```

3. Publish the first version (cloning an unknown package name creates it):

   ```bash
   sudo apt install makepkg              # for .SRCINFO, on Ubuntu
   packaging/aur/update.sh 1.10.0         # or the latest released version
   packaging/aur/publish.sh
   ```

   Check https://aur.archlinux.org/packages/python-voiceio, then on an Arch
   machine: `yay -S python-voiceio && voiceio --version`.

4. Let CI do the rest: add the private key as a repository secret,

   ```bash
   gh secret set AUR_SSH_PRIVATE_KEY < ~/.ssh/aur
   ```

   From then on every tag updates the AUR. Without the secret, the release
   still gets the updated `PKGBUILD` and `SRCINFO` attached; run step 3's
   last two commands to push them.

A packaging-only fix (same version): `packaging/aur/update.sh X.Y.Z 2` then
`publish.sh`.
