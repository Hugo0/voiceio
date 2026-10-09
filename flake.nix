# voiceio flake.
#
# UNBUILT: this file was written without a Nix toolchain and has never been
# evaluated or built. Treat it as a starting point and please report fixes.
#
#   nix run github:Hugo0/voiceio -- setup
#   nixosModules.default       -> programs.voiceio.enable (udev rule, uinput)
#   homeManagerModules.default -> services.voiceio (config.toml + user unit)
{
  description = "voiceio: local voice dictation for Linux";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      packages = forAllSystems (pkgs: {
        voiceio = pkgs.callPackage ./nix/package.nix { };
        default = self.packages.${pkgs.stdenv.hostPlatform.system}.voiceio;
      });

      nixosModules.default = import ./nix/nixos-module.nix self;
      homeManagerModules.default = import ./nix/hm-module.nix self;

      checks = forAllSystems (pkgs: {
        pytest = self.packages.${pkgs.stdenv.hostPlatform.system}.voiceio.overridePythonAttrs {
          doCheck = true;
        };
      });

      formatter = forAllSystems (pkgs: pkgs.nixfmt-rfc-style);
    };
}
