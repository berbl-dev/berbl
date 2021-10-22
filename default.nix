{ lib, buildPythonPackage } :

buildPythonPackage rec {
  pname = "berbl";
  version = "0.1.0";

  src = ./.;

  doCheck = false;

  meta = with lib; {
    homepage = "https://github.com/dpaetzel/berbl";
    description = "Implementation of a Bayesian Evolutionary Rule-based Learner";
    license = licenses.gpl3;
    maintainers = with maintainers; [ dpaetzel ];
  };
}
