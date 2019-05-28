#!/bin/bash
set -ex
python3 -m flake8 /ci/
(cd / && python3 -m pylint --rcfile /pylintrc ci --score=n)
