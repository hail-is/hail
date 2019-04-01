#!/bin/bash
set -ex

python3 -m pip install -r ./requirements.txt

make test-in-cluster
