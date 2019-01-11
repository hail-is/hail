#!/bin/bash
set -ex

flake8 pipeline
pylint pipeline --rcfile pipeline/pylintrc --score=n
pytest test
