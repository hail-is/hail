#!/bin/bash

apt -y install git emacs-nox
python3 -m pip install nest_asyncio python-json-logger

git clone https://github.com/cseed/hail.git
(cd hail && git checkout hack)
