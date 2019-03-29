#!/bin/bash
set -ex

wd=$cwd
cd scripts/users && ./hail-ci-build.sh && cd $wd