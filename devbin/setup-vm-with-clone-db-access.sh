#!/bin/bash

set -ex

apt-get update
apt-get install -y emacs-nox git mysql-client python3-pip

mkdir /sql-config/
chmod 777 /sql-config/

git clone https://github.com/hail-is/hail.git
cd hail

pip3 install -r hail/python/requirements.txt

mkdir /gear/
chmod 777 /gear/
cp gear/setup.py /gear/
cp -R gear/gear /gear/
pip3 install /gear

mkdir /hailtop/
chmod 777 /hailtop/
cp hail/python/setup-hailtop.py /hailtop/setup.py
cp -R hail/python/hailtop /hailtop/
echo "0.2.111" > /hail_version
cp /hail_version /hailtop/hailtop/hail_version
cp hail/python/MANIFEST.in /hailtop/MANIFEST.in
pip3 install /hailtop
