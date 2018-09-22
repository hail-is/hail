#!/bin/bash
set -ex

(cd batch && /bin/bash hail-ci-deploy.sh)
