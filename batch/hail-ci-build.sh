#!/bin/sh
set -ex

flake8 batch
PYLINT_OUTPUT_FILE=$(mktemp)
pylint batch --rcfile batch/pylintrc >$PYLINT_OUTPUT_FILE --score=n || true
diff $PYLINT_OUTPUT_FILE ignored-pylint-errors
make test-local-in-cluster
