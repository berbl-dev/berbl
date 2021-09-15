with import (builtins.fetchGit {
  name = "nixpkgs-2021-09-14";
  url = "https://github.com/NixOS/nixpkgs/";
  rev = "03e61e35dd911a43690342d4721f3c964d1bd621";
}) {
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
          doInstallCheck = false;
        });
        mlflowPatched = (python-super.mlflow.override {
          sqlalchemy = python-self.sqlalchemy;
          # requires an older version of alembic
          alembic = python-self.alembic;
        }).overrideAttrs (attrs: {
          propagatedBuildInputs = attrs.propagatedBuildInputs
            ++ (with python-self; [
              prometheus-flask-exporter
              azure-storage-blob
            ]);
          meta.broken = false;
        });
      };
    };
  };
};

stdenv.mkDerivation rec {
  name = "prolcs";

  buildInputs = [
    (python3.withPackages (ps:
      with ps; [
        # TODO azure-storage-blob is only required because otherwise mlflow
        # doesn't find it at runtime (whereas ipython does, actually?!).
        azure-storage-blob
        click
        deap
        hypothesis
        matplotlib
        mlflowPatched
        numpy
        pandas
        scipy
        scikitlearn
        seaborn
        pytest
      ]))
    mypy
    python3Packages.mlflowPatched
    python3Packages.ipython
    python3Packages.pytest
    python3Packages.sphinx
  ];
}

# 1. Do not use tox (yet, because it doesn't work with scipy requiring
# libstdc++6).
#     [2020-09-24] tox gives me problems because scipy(?) requires not only
#     Python packages but also libstdcxx5, which needs to be installed via Nix
#     (and I didn't find out in acceptable time how to have tox find site
#     packages/libraries)
#     python3Packages.tox
# 2. Run tests by: PYTHONPATH=src pytest test
