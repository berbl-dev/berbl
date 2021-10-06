(let
  bootstrap = import <nixpkgs> { };
  pkgs = import (bootstrap.fetchFromGitHub {
    owner = "NixOS";
    repo = "nixpkgs";
    rev = "f18bd6f9192e0f4b78419efe4caaf0fd37a0a346";
    sha256 = "0d0m25xxl9s6ljk27x9308zqawdbmhd7m9cwkcajsbn684imcw85";
  }) { };
  prolcs = pkgs.callPackage ./default.nix {
    buildPythonPackage = pkgs.python38Packages.buildPythonPackage;
  };
in
  with pkgs;
  python38.withPackages (ps: with ps; [
    deap
    hypothesis # actually just a test dependency
    numpy
    pandas
    prolcs
    pytest # actually just a test dependency
    scipy
    scikitlearn
  ])
).env
