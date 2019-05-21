#!/bin/bash
set -ex
python3 -m flake8 /ci2/
python3 -m pylint --rcfile /pylintrc /ci2/ --score=n
