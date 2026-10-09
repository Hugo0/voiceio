# voiceio built from this checkout.
#
# UNBUILT: written without a Nix toolchain to test against. The runtime
# wrapping follows NostraDavid's nixcfg derivation (pkgs/voiceio), minus its
# substituteInPlace patches: voiceio now honours $VOICEIO_SYSTEM_PYTHON
# (voiceio/platform.py find_system_python) for the GObject subprocesses.
{
  lib,
  python3,
  wrapGAppsHook3,
  gobject-introspection,
  at-spi2-core,
  gdk-pixbuf,
  gtk3,
  harfbuzz,
  pango,
  ibus,
  libayatana-appindicator,
  libnotify,
  procps,
  wl-clipboard,
  wtype,
  xclip,
  xdotool,
  ydotool,
}:
let
  py = python3.pkgs;
  pyproject = builtins.fromTOML (builtins.readFile ../pyproject.toml);

  # Base + `desktop` extra. `server`/`tts`/`train`/`sherpa` stay opt-in.
  runtimeDeps = with py; [
    faster-whisper
    numpy
    onnxruntime
    wordfreq
    sounddevice
    evdev
    pynput
    pystray
    pillow
  ];

  # The IBus engine and the AppIndicator tray run under an interpreter that
  # has PyGObject; voiceio finds it through VOICEIO_SYSTEM_PYTHON.
  systemPython = python3.withPackages (ps: runtimeDeps ++ [ ps.pygobject3 ]);
in
py.buildPythonApplication {
  pname = "python-voiceio";
  inherit (pyproject.project) version;
  pyproject = true;

  src = lib.cleanSource ../.;

  build-system = with py; [
    setuptools
    wheel
  ];

  dependencies = runtimeDeps;

  nativeBuildInputs = [
    wrapGAppsHook3
    gobject-introspection
  ];

  buildInputs = [
    gtk3
    ibus
    libayatana-appindicator
  ];

  # Fold the GApps environment into the Python wrapper (see preFixup)
  # instead of wrapping every program twice.
  dontWrapGApps = true;

  pythonImportsCheck = [
    "voiceio"
    "voiceio.core.engine"
  ];

  # Tests run in the flake's `checks.pytest` (see flake.nix), not on every build.
  doCheck = false;
  nativeCheckInputs = with py; [
    pytestCheckHook
    pytest-mock
    aiohttp
  ];
  preCheck = ''
    export HOME=$(mktemp -d)
  '';

  # One wrapper per program: buildPythonApplication's own wrapper picks up
  # makeWrapperArgs, and the GApps hook's args are appended to it.
  makeWrapperArgs = [
    "--prefix"
    "PATH"
    ":"
    (lib.makeBinPath [
      ibus
      libnotify
      procps
      wl-clipboard
      wtype
      xclip
      xdotool
      ydotool
    ])
    "--prefix"
    "GI_TYPELIB_PATH"
    ":"
    (lib.makeSearchPath "lib/girepository-1.0" [
      at-spi2-core
      gdk-pixbuf
      gobject-introspection
      gtk3
      harfbuzz
      ibus
      libayatana-appindicator
      pango.out
    ])
    "--set"
    "VOICEIO_SYSTEM_PYTHON"
    "${systemPython}/bin/python3"
  ];

  preFixup = ''
    makeWrapperArgs+=("''${gappsWrapperArgs[@]}")
  '';

  passthru = { inherit systemPython; };

  meta = {
    description = pyproject.project.description;
    homepage = "https://github.com/Hugo0/voiceio";
    license = lib.licenses.mit;
    mainProgram = "voiceio";
    platforms = lib.platforms.linux;
  };
}
