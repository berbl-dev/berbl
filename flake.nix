{
  description = "The berbl Python library";

  inputs = {
    nixos-config.url = "github:dpaetzel/nixos-config";

    overlays.url = "github:dpaetzel/overlays";
    overlays.inputs.nixpkgs.follows = "nixos-config/nixpkgs";

    mkdocstringsSrc.url = "github:mkdocstrings/mkdocstrings/0.18.0";
    mkdocstringsSrc.flake = false;

    mkdocstringsPythonLegacySrc.url = "github:mkdocstrings/python-legacy/0.2.2";
    mkdocstringsPythonLegacySrc.flake = false;

    mkdocstringsPythonSrc.url = "github:mkdocstrings/python/0.6.6";
    mkdocstringsPythonSrc.flake = false;

    pytkdocsSrc.url = "github:mkdocstrings/pytkdocs/0.16.1";
    pytkdocsSrc.flake = false;

    mkdocsAutorefsSrc.url = "github:mkdocstrings/autorefs/0.4.1";
    mkdocsAutorefsSrc.flake = false;

    mkdocsGenFilesSrc.url = "github:oprypin/mkdocs-gen-files/v0.3.4";
    mkdocsGenFilesSrc.flake = false;

    mkdocsLiterateNavSrc.url = "github:oprypin/mkdocs-literate-nav/v0.4.1";
    mkdocsLiterateNavSrc.flake = false;

    mkdocsSectionIndexSrc.url = "github:oprypin/mkdocs-section-index/v0.3.4";
    mkdocsSectionIndexSrc.flake = false;

    griffeSrc.url = "github:mkdocstrings/griffe/0.16.0";
    griffeSrc.flake = false;
  };

  outputs = inputs@{ self, nixos-config, overlays, ... }:

    let
      nixpkgs = nixos-config.inputs.nixpkgs;
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = with overlays.overlays; [ mlflow ];
      };
      python = pkgs.python39;

      griffe = python.pkgs.buildPythonPackage rec {
        pname = "griffe";
        version = "0.15.0";

        src = inputs.griffeSrc;

        format = "pyproject";

        # Dynamically getting version via pdm doesn't seem to work.
        postPatch = ''
          sed -i 's/dynamic = \[\"version\"\]/version = \"${version}\"/' pyproject.toml
          sed -i 's/^version.*use_scm.*$//' pyproject.toml
        '';

        nativeBuildInputs = [ python.pkgs.pdm-pep517 ];

        propagatedBuildInputs = with python.pkgs; [ cached-property ];

        doCheck = false;
      };

      mkdocs-section-index = python.pkgs.buildPythonPackage rec {
        pname = "mkdocs-section-index";
        version = "0.3.4";

        src = inputs.mkdocsSectionIndexSrc;

        format = "pyproject";

        nativeBuildInputs = [ python.pkgs.poetry ];

        propagatedBuildInputs = with python.pkgs; [ mkdocs ];

        doCheck = false;
      };

      mkdocs-literate-nav = python.pkgs.buildPythonPackage rec {
        pname = "mkdocs-literate-nav";
        version = "0.4.1";

        src = inputs.mkdocsLiterateNavSrc;

        format = "pyproject";

        nativeBuildInputs = [ python.pkgs.poetry ];

        propagatedBuildInputs = with python.pkgs; [ mkdocs ];

        doCheck = false;
      };

      mkdocs-gen-files = python.pkgs.buildPythonPackage rec {
        pname = "mkdocs-gen-files";
        version = "0.3.4";

        src = inputs.mkdocsGenFilesSrc;

        format = "pyproject";

        nativeBuildInputs = [ python.pkgs.poetry ];

        propagatedBuildInputs = with python.pkgs; [ mkdocs ];

        doCheck = false;

      };

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

          # For parsing NumPy docstrings.
          docstring-parser
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

      mkdocstrings-python = python.pkgs.buildPythonPackage rec {
        pname = "mkdocstrings-python";
        version = "0.6.6";

        src = inputs.mkdocstringsPythonSrc;

        format = "pyproject";

        # Dynamically getting version via pdm doesn't seem to work.
        postPatch = ''
          sed -i 's/dynamic = \[\"version\"\]/version = \"${version}\"/' pyproject.toml
          sed -i 's/^version.*use_scm.*$//' pyproject.toml

          # There seems to be a circular dependency here.
          sed -i "s/^.*\"mkdocstrings>=.*$//" pyproject.toml
        '';

        nativeBuildInputs = [ python.pkgs.pdm-pep517 ];

        propagatedBuildInputs = [ griffe ];

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
          mkdocstrings-python
          mkdocs-autorefs
        ];

        doCheck = false;

      };
    in rec {
      defaultPackage.${system} = python.pkgs.buildPythonPackage rec {
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
        ];

        testInputs = with python.pkgs; [ hypothesis pytest ];

        doCheck = false;

        meta = with pkgs.lib; {
          description =
            "Implementation of a Bayesian Learning Classifier System";
          license = licenses.gpl3;
        };
      };

      devShell.${system} = pkgs.mkShell {
        packages = [
          pkgs.mkdocs
          mkdocstrings
          mkdocs-gen-files
          mkdocs-literate-nav
          mkdocs-section-index
        ] ++ (with python.pkgs; [
          tox
          mkdocs-material-extensions
          mkdocs-material
        ]) ++ defaultPackage.${system}.testInputs
          ++ defaultPackage.${system}.propagatedBuildInputs;
      };
    };
}
