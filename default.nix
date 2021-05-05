with import (builtins.fetchGit {
  name = "nixpkgs-2020-10-20";
  url = "https://github.com/NixOS/nixpkgs/";
  rev = "97d0eb5059f6b6994c37402711455365bdbb63d4";
}) {
  config.packageOverrides = super: {
    python3 = super.python3.override {
      packageOverrides = python-self: python-super: {
        prometheus_flask_exporter = python-super.buildPythonPackage rec {
          pname = "prometheus_flask_exporter";
          version = "0.15.4";

          src = python-super.fetchPypi {
            inherit pname version;
            sha256 =
              "c590656b45fa6dd23d81dec3d3dc1e31b17fcba48310f69d0ff31b5c865fc799";
          };

          propagatedBuildInputs = with python-super; [
            flask
            prometheus_client
          ];

          meta = with super.stdenv.lib; {
            homepage = "https://github.com/rycus86/prometheus_flask_exporter";
            description = "Prometheus exporter for Flask applications";
            maintainers = with maintainers; [ nphilou ];
            license = licenses.mit;
          };
        };
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
        mlflowPatched = (python-super.mlflow.override {
          sqlalchemy = python-self.sqlalchemy;
        }).overrideAttrs (attrs: {
          propagatedBuildInputs = attrs.propagatedBuildInputs
            ++ (with python-self; [
              prometheus_flask_exporter
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
    # [2020-09-24] tox gives me problems because scipy(?) requires not only
    # Python packages but also libstdcxx5, which needs to be installed via Nix
    # (and I didn't find out in acceptable time how to have tox find site
    # packages/libraries)
    # python3Packages.tox
  ];
}

# 1. Do not use tox (yet, because it doesn't work with scipy requiring
# libstdc++6).
# 2. Run tests by: PYTHONPATH=src pytest test
