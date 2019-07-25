#!/bin/bash
set -ex

apt -y update
apt -y dist-upgrade
apt -y install docker.io python3-pip
pip3 install aiohttp

shutdown -h now
