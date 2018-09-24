#!/bin/bash
set -ex

(cd hail && /bin/bash hail-ci-deploy.sh)
(cd batch && /bin/bash hail-ci-deploy.sh)
(cd site && /bin/bash hail-ci-deploy.sh)
