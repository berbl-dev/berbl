with import /home/david/Code/nixpkgs {};

stdenv.mkDerivation rec {
  name = "prolcs";

  buildInputs = [
    (python3.withPackages (ps: with ps; [
      click
      hypothesis
      matplotlib
      numpy
      pandas
      # sacred
      scipy
      scikitlearn
      seaborn
      pytest
    ]))
    mypy
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
