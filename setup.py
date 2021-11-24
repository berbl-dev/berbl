#!/usr/bin/env python

from distutils.core import setup

setup(
    name="berbl",
    version="0.1",
    description="Bayesian Evolutionary Rule-based Learner",
    author="berbl",
    author_email="berbl@berbl.berbl",
    url="https://github.com/tmp-berbl/berbl",
    packages=[
        "berbl", "berbl.literal", "berbl.search", "berbl.search.ga",
        "berbl.search.operators", "berbl.match"
    ],
    package_dir={"": "src"},
    install_requires=[
        "deap ==1.3.1",
        "mlflow ==1.20.2",
        "numpy ==1.21.2",
        "scipy ==1.7.1",
        "scikit-learn ==0.24.1",
    ],
    tests_requires=[
        "hypothesis",
        "pytest"
    ])
