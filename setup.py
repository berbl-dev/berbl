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
        "prolcs.search.operators"
    ],
    package_dir={"": "src"},
)
