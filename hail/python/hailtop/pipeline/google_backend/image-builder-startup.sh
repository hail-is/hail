#!/bin/bash
set -ex

apt -y update
apt -y dist-upgrade
apt -y install emacs-nox docker.io python3-pip
python3 -m pip install python-json-logger aiohttp sortedcontainers google-api-python-client google-cloud uvloop

shutdown -h now
