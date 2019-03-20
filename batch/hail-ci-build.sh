#!/bin/bash
set -ex

flake8 batch
pylint batch --rcfile batch/pylintrc --score=n
make test-local-in-cluster
