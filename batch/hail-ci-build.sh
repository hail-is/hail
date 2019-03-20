#!/bin/bash
set -ex

. ../loadconda
conda activate hail-batch

make test-local-in-cluster
