#!/bin/bash
set -ex
python3 -m flake8 /batch/
python3 -m pylint --rcfile /pylintrc /batch/ --score=n
python3 -m pytest -vv -s /test/
