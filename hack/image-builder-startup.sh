#!/bin/bash
set -ex

apt -y update
apt -y dist-upgrade
apt -y install docker.io python3-pip
pip3 install aiohttp sortedcontainers google-api-python-client google-cloud==0.32.0 uvloop

shutdown -h now
