#!/bin/sh

set -e

# clean directories
rm -rf build/tmp/doctest

# make directories
mkdir -p build/tmp/doctest
cp -r python/ build/tmp/doctest/python/

cd build/tmp/doctest/python/hail
pytest -n $PARALLELISM --dist=loadscope --doctest-modules --ignore=docs/conf.py --ignore=tests --ignore=setup.py
