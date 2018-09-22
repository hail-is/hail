#!/bin/bash
set -ex

(cd batch && /bin/bash hail-ci-deploy.sh)
(cd hail && /bin/bash hail-ci-deploy.sh)
