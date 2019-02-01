#!/bin/bash
set -ex

. ../loadconda
conda activate hail-pipeline

make test-local-in-cluster

#flake8 pipeline
#pylint pipeline --rcfile pipeline/pylintrc --score=n
#pytest -v test
