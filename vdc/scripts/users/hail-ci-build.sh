#!/bin/bash
set -ex

python3 -m pip install --user -r ./requirements.txt

make test-in-cluster
