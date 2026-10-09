# NixOS module: `programs.voiceio.enable = true;`
#
# UNBUILT (no Nix toolchain was available when this was written).
self:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.programs.voiceio;

  # Same rule `voiceio doctor --fix` installs on other distros: the active
  # seat user may read keyboard event devices (evdev hotkeys), no `input`
  # group or re-login needed.
  uaccessRules = pkgs.writeTextFile {
    name = "voiceio-udev-rules";
    destination = "/lib/udev/rules.d/60-voiceio-uaccess.rules";
    text = builtins.readFile ../voiceio/data/60-voiceio-uaccess.rules;
  };
in
{
  options.programs.voiceio = {
    enable = lib.mkEnableOption "voiceio voice dictation";

    package = lib.mkOption {
      type = lib.types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      defaultText = lib.literalExpression "voiceio.packages.\${system}.default";
      description = "The voiceio package to install.";
    };
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ cfg.package ];

    # /dev/uinput for the ydotool typer. Users that type through ydotool also
    # need to be in the `uinput` group (users.users.<name>.extraGroups).
    hardware.uinput.enable = true;

    services.udev.packages = [ uaccessRules ];

    # Streaming preedit on Wayland needs IBus as the input method; enable it
    # yourself if you want it (fcitx5 users fall back to wtype/ydotool):
    #   i18n.inputMethod = { enable = true; type = "ibus"; };
  };
}
