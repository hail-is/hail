#!/bin/bash
set -ex

(cd hail && /bin/bash hail-ci-deploy.sh)
(cd batch && /bin/bash hail-ci-deploy.sh)
