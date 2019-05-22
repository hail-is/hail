#!/bin/bash
set -ex
python3 -m flake8 /batch/batch
python3 -m pylint --rcfile /pylintrc /batch/batch --score=n
python3 -m pytest -vv -s /test/
