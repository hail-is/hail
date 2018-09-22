#!/bin/bash
set -ex

(cd batch && /bin/bash hail-ci-build.sh)
