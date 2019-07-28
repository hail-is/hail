#!/bin/bash

apt -y update
apt -y dist-upgrade
apt -y install git emacs-nox
python3 -m pip install nest_asyncio python-json-logger sortedcontainers google-api-python-client google-cloud==0.32.0

git clone https://github.com/cseed/hail.git
(cd hail && git checkout hack)
