{ lib, buildPythonPackage, deap, mlflow, numpy, numpydoc, pandas, scipy
, scikitlearn, hypothesis, pytest, sphinx }:

buildPythonPackage rec {
  pname = "berbl";
  version = "0.1.0";

  src = ./.;

  propagatedBuildInputs =
    [ deap mlflow numpy numpydoc pandas scipy scikitlearn sphinx ];

  testInputs = [ hypothesis pytest ];

  doCheck = false;

  meta = with lib; {
    description =
      "Implementation of a Bayesian Learning Classifier System";
    license = licenses.gpl3;
  };
}
