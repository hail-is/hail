#!/bin/bash
set -ex

. ../loadconda
conda activate hail-pipeline

make test-local-in-cluster
