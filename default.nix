{ lib, buildPythonPackage, deap, mlflow, numpy, pandas, scipy, scikitlearn
, hypothesis, pytest }:

buildPythonPackage rec {
  pname = "berbl";
  version = "0.1.0";

  src = ./.;

  propagatedBuildInputs = [ deap mlflow numpy pandas scipy scikitlearn ];

  testInputs = [ hypothesis pytest ];

  doCheck = false;

  meta = with lib; {
    homepage = "https://github.com/dpaetzel/berbl";
    description =
      "Implementation of a Bayesian Evolutionary Rule-based Learner";
    license = licenses.gpl3;
    maintainers = with maintainers; [ dpaetzel ];
  };
}
