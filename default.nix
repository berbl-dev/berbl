{ lib, buildPythonPackage } :

buildPythonPackage rec {
  pname = "prolcs";
  version = "0.1.0";

  src = ./.;

  doCheck = false;

  meta = with lib; {
    homepage = "https://github.com/dpaetzel/prolcs";
    description = "Implementation of a fully Bayesian LCS";
    license = licenses.gpl3;
    maintainers = with maintainers; [ dpaetzel ];
  };
}
