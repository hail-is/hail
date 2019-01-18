#!/bin/bash
set -ex

. ../loadconda
conda activate hail-pipeline

flake8 pipeline
pylint pipeline --rcfile pipeline/pylintrc --score=n
pytest -v test
