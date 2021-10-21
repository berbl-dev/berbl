#!/usr/bin/env python

from distutils.core import setup

setup(
    name="prolcs",
    version="0.1",
    description="Probabilistic LCS",
    author="David Pätzel",
    author_email="david.paetzel@posteo.de",
    url="https://github.com/dpaetzel/prolcs",
    packages=[
        "prolcs", "prolcs.literal", "prolcs.search", "prolcs.search.ga",
        "prolcs.search.operators", "prolcs.match"
    ],
    package_dir={"": "src"},
    install_requires=[
        "deap ==1.3.1",
        "mlflow ==1.20.1",
        "numpy ==1.21.2",
        "pandas ==1.3.2",
        "scipy ==1.7.1",
        "scikit-learn ==0.24.1",

        # just required for testing
        "hypothesis",
        "pytest"
    ])
