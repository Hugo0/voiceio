# Home Manager module: `services.voiceio = { enable; package; settings; }`.
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
  cfg = config.services.voiceio;
  tomlFormat = pkgs.formats.toml { };
in
{
  options.services.voiceio = {
    enable = lib.mkEnableOption "voiceio voice dictation daemon";

    package = lib.mkOption {
      type = lib.types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      defaultText = lib.literalExpression "voiceio.packages.\${system}.default";
      description = "The voiceio package to run.";
    };

    settings = lib.mkOption {
      inherit (tomlFormat) type;
      default = { };
      example = lib.literalExpression ''
        {
          model.name = "small";
          hotkey.key = "ctrl+alt+v";
          output.method = "auto";
        }
      '';
      description = ''
        Written to {file}`~/.config/voiceio/config.toml`. See
        config.example.toml in the voiceio repository for every key.
        When set, the file is a read-only store path, so `voiceio setup`
        and `voiceio config` cannot edit it; leave it empty to manage the
        config imperatively.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    home.packages = [ cfg.package ];

    xdg.configFile."voiceio/config.toml" = lib.mkIf (cfg.settings != { }) {
      source = tomlFormat.generate "voiceio-config.toml" cfg.settings;
    };

    # Mirrors the unit voiceio/service.py writes, pointed at the store path.
    systemd.user.services.voiceio = {
      Unit = {
        Description = "VoiceIO - voice-to-text input";
        Documentation = "https://github.com/Hugo0/voiceio";
        After = [ "graphical-session.target" ];
        PartOf = [ "graphical-session.target" ];
      };
      Service = {
        Type = "simple";
        # `run` is the v1.0 daemon subcommand (bare `voiceio` before it).
        ExecStart = "${cfg.package}/bin/voiceio run";
        Nice = 10;
        Restart = "on-failure";
        RestartSec = 3;
        Environment = [ "PYTHONUNBUFFERED=1" ];
        PassEnvironment = [
          "DISPLAY"
          "WAYLAND_DISPLAY"
          "XDG_SESSION_TYPE"
          "XDG_RUNTIME_DIR"
        ];
      };
      Install.WantedBy = [ "graphical-session.target" ];
    };
  };
}
