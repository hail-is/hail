#!/bin/bash
set -ex

pip3 install -r ./requirements.txt

make test-in-cluster
