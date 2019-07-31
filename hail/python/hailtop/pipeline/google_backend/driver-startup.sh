#!/bin/bash

apt -y update
apt -y dist-upgrade
apt -y install git emacs-nox docker.io python3-pip
python3 -m pip install nest_asyncio aiohttp python-json-logger sortedcontainers google-api-python-client google-cloud uvloop

git clone https://github.com/cseed/hail.git
(cd hail && git checkout hack)
