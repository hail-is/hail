#!/bin/bash
set -ex

apt -y update
apt -y dist-upgrade
apt -y install emacs-nox docker.io python3-pip
python3 -m pip install python-json-logger gcsfs==0.2.1 hurry.filesize==0.9 aiohttp sortedcontainers google-api-python-client google-cloud google-cloud-logging uvloop

shutdown -h now
