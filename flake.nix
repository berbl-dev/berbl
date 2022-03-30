{
  description = "The berbl Python library";

  inputs.nixpkgs.url =
    # 2022-03-29
    "github:NixOS/nixpkgs/0e3d0d844e89da74081f0e77c1da36a2eb3a8ff7";

  inputs.overlays.url = "github:dpaetzel/overlays";

  inputs.mkdocstringsSrc.url = "github:mkdocstrings/mkdocstrings/0.18.0";
  inputs.mkdocstringsSrc.flake = false;

  inputs.mkdocstringsPythonLegacySrc.url =
    "github:mkdocstrings/python-legacy/0.2.2";
  inputs.mkdocstringsPythonLegacySrc.flake = false;

  inputs.pytkdocsSrc.url = "github:mkdocstrings/pytkdocs/0.16.1";
  inputs.pytkdocsSrc.flake = false;

  inputs.mkdocsAutorefsSrc.url = "github:mkdocstrings/autorefs/0.4.1";
  inputs.mkdocsAutorefsSrc.flake = false;

  outputs = inputs@{ self, nixpkgs, overlays, ... }:

    with import nixpkgs {
      system = "x86_64-linux";
      overlays = with overlays.overlays; [ mlflow ];
    };
    let
      python = python39;
      pytkdocs = python.pkgs.buildPythonPackage rec {
        pname = "pytkdocs";
        version = "0.16.1";

        src = inputs.pytkdocsSrc;

        format = "pyproject";

        # Dynamically getting version via pdm doesn't seem to work.
        postPatch = ''
          sed -i 's/dynamic = \[\"version\"\]/version = \"${version}\"/' pyproject.toml
          sed -i 's/^version.*use_scm.*$//' pyproject.toml
        '';

        nativeBuildInputs = [ python.pkgs.pdm-pep517 ];

        propagatedBuildInputs = with python.pkgs; [
          astunparse
          cached-property
          typing-extensions
        ];

        doCheck = false;

      };
      mkdocstrings-python-legacy = python.pkgs.buildPythonPackage rec {
        pname = "mkdocstrings-python-legacy";
        version = "0.2.2";

        src = inputs.mkdocstringsPythonLegacySrc;

        format = "pyproject";

        # Dynamically getting version via pdm doesn't seem to work.
        postPatch = ''
          sed -i 's/dynamic = \[\"version\"\]/version = \"${version}\"/' pyproject.toml
          sed -i 's/^version.*use_scm.*$//' pyproject.toml

          # There seems to be a circular dependency here.
          sed -i "s/^.*\"mkdocstrings>=.*$//" pyproject.toml
        '';

        nativeBuildInputs = [ python.pkgs.pdm-pep517 ];

        propagatedBuildInputs = [ pytkdocs ];

        doCheck = false;

      };
      mkdocs-autorefs = python.pkgs.buildPythonPackage rec {
        pname = "mkdocs-autorefs";

        version = "0.4.1";

        src = inputs.mkdocsAutorefsSrc;

        format = "pyproject";

        # Dynamically getting version via pdm doesn't seem to work.
        postPatch = ''
          sed -i 's/dynamic = \[\"version\"\]/version = \"${version}\"/' pyproject.toml
          sed -i 's/^version.*use_scm.*$//' pyproject.toml
        '';

        nativeBuildInputs = [ python.pkgs.pdm-pep517 ];

        propagatedBuildInputs = with python.pkgs; [ markdown mkdocs pytkdocs ];

        doCheck = false;

      };
      mkdocstrings = python.pkgs.buildPythonPackage rec {
        pname = "mkdocstrings";
        version = "0.18.0";

        src = inputs.mkdocstringsSrc;

        format = "pyproject";

        # Dynamically getting version via pdm doesn't seem to work.
        postPatch = ''
          sed -i 's/dynamic = \[\"version\"\]/version = \"${version}\"/' pyproject.toml
          sed -i 's/^version.*use_scm.*$//' pyproject.toml
        '';

        nativeBuildInputs = [ python.pkgs.pdm-pep517 ];

        propagatedBuildInputs = with python.pkgs; [
          mkdocs
          jinja2
          markdown
          markupsafe
          pymdown-extensions
          mkdocstrings-python-legacy
          mkdocs-autorefs
        ];

        doCheck = false;

      };
    in rec {
      defaultPackage.x86_64-linux = python.pkgs.buildPythonPackage rec {
        pname = "berbl";
        version = "0.1.0";

        src = self;

        # We use pyproject.toml.
        format = "pyproject";

        propagatedBuildInputs = with python.pkgs; [
          deap
          mlflow
          numpy
          numpydoc
          pandas
          scipy
          scikitlearn
          sphinx
        ];

        testInputs = with python.pkgs; [ hypothesis pytest ];

        doCheck = false;

        meta = with lib; {
          description =
            "Implementation of a Bayesian Learning Classifier System";
          license = licenses.gpl3;
        };
      };

      devShell.x86_64-linux = mkShell {
        inherit mkdocstrings;
        packages =
          [ defaultPackage.x86_64-linux python.pkgs.tox mkdocs mkdocstrings ]
          ++ defaultPackage.x86_64-linux.testInputs;
      };
    };
}
