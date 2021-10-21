let
  bootstrap = import <nixpkgs> { };
  pkgs = import (bootstrap.fetchFromGitHub {
    owner = "NixOS";
    repo = "nixpkgs";
    rev = "f18bd6f9192e0f4b78419efe4caaf0fd37a0a346";
    sha256 = "0d0m25xxl9s6ljk27x9308zqawdbmhd7m9cwkcajsbn684imcw85";
  }) {
    # TODO Extract this override to an mlflow.nix and reuse that in the
    # experiments project
    config.packageOverrides = super: {
      python3 = super.python3.override {
        packageOverrides = python-self: python-super: {
          sqlalchemy = python-super.sqlalchemy.overrideAttrs (attrs: rec {
            pname = "SQLAlchemy";
            version = "1.3.13";
            src = python-super.fetchPypi {
              inherit pname version;
              sha256 =
                "sha256:1yxlswgb3h15ra8849vx2a4kp80jza9hk0lngs026r6v8qcbg9v4";
            };
            doInstallCheck = false;
          });
          alembic = python-super.alembic.overrideAttrs (attrs: rec {
            pname = "alembic";
            version = "1.4.1";
            src = python-super.fetchPypi {
              inherit pname version;
              sha256 =
                "sha256:0a4hzn76csgbf1px4f5vfm256byvjrqkgi9869nkcjrwjn35c6kr";
            };
            propagatedBuildInputs = with python-super; [
              python-editor
              python-dateutil
              python-self.sqlalchemy
              Mako
            ];
            doInstallCheck = false;
          });
          mlflowPatched = (python-super.mlflow.override {
            sqlalchemy = python-self.sqlalchemy;
            # requires an older version of alembic
            alembic = python-self.alembic;
          }).overrideAttrs (attrs: {
            propagatedBuildInputs = attrs.propagatedBuildInputs
              ++ (with python-self; [
                importlib-metadata
                prometheus-flask-exporter
                azure-storage-blob
              ]);
            meta.broken = false;
          });
        };
      };
    };
  };
  prolcs = pkgs.callPackage ./default.nix {
    buildPythonPackage = pkgs.python38Packages.buildPythonPackage;
  };
in pkgs.mkShell rec {
  name = "piure";
  packages = with pkgs; [
    (python3.withPackages (ps: with ps; [
      deap
      mlflowPatched
      numpy
      pandas
      prolcs
      scipy
      scikitlearn

      # test dependencies
      hypothesis
      pytest
      tox
    ]))
  ];
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
  '';
}
