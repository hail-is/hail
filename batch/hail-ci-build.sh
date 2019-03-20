#!/bin/bash
set -ex

conda activate hail-batch

make test-local-in-cluster
