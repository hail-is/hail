#!/bin/bash
set -ex

(cd hail && /bin/bash hail-ci-build.sh)
(cd batch && /bin/bash hail-ci-build.sh)
