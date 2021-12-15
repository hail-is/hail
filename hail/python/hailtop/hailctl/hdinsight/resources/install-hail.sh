#!/bin/bash

hail_pip_version=$1

apt-get install -y \
    openjdk-8-jre-headless \
    g++ \
    python3.6 python3-pip \
    libopenblas-base liblapack3

/usr/bin/anaconda/bin/conda create -n py37 python=3.7 --yes
/usr/bin/anaconda/bin/conda activate py37
# this installs a bunch of jupyter dependencies...
# python3 -m pip install 'https://github.com/hail-is/jgscm/archive/v0.1.12+hail.zip'
python3 -m pip install "hail==$hail_pip_version" --no-dependencies
site_packages=$(python3 -m pip show hail| grep -E 'Location:' | sed -E 's/^Location: //')
# https://www.python.org/dev/peps/pep-0440/#version-specifiers
grep -E '^Requires-Dist:' $site_packages/hail-$hail_pip_version.dist-info/METADATA \
    | sed 's/Requires-Dist: //' \
    | sed -E 's/ \(([^)]*)\)/\1/' \
    | grep -Ev '^pyspark(~=|==|!=|<=|>=|<|>|===|$)' \
    > requirements.txt
python3 -m pip install -r requirements.txt
