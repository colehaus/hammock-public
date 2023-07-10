{
  description = "Application packaged using poetry2nix";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.poetry2nix = {
    url = "github:nix-community/poetry2nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        inherit (poetry2nix.legacyPackages.${system})
          mkPoetryApplication mkPoetryEnv overrides;
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python311;
        myOverrides = overrides.withDefaults (final: prev:
          # Missing setup tools dependency declarations
          (pkgs.lib.genAttrs [
            "httpsproxy-urllib2"
            "instructorembedding"
            "wikipedia"
          ] (name:
            prev.${name}.overridePythonAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ prev.setuptools ];
            }))) //
          # Miscellaneous build problems that are most easily fixed by using wheels
          (pkgs.lib.genAttrs [
            "cmake"
            "ruff"
            "safetensors"
            "tokenizers"
            "pybind11"
            "scipy"
            "urllib3"
          ] (name: prev.${name}.override { preferWheel = true; })));
        poetryAttrs = {
          projectDir = ./.;
          preferWheels = false;
          python = python;
          overrides = myOverrides;
        };
      in rec {
        formatter = pkgs.nixfmt;
        defaultApp = mkPoetryApplication poetryAttrs;
        devShells.default = (mkPoetryEnv poetryAttrs).env.overrideAttrs
          (final: prev: {
            nativeBuildInputs = (prev.nativeBuildInputs or [ ]) ++ [
              poetry2nix.packages.${system}.poetry
              pkgs.typescript
              pkgs.nodePackages.prettier
            ];
          });
      });
}
