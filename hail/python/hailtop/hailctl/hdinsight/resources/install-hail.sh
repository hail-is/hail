#!/bin/bash

set -ex

hail_pip_specifier=$1
hail_pip_version=$2
cluster_name=$3

/usr/bin/anaconda/bin/conda create -n py37 python=3.7 --yes
source /usr/bin/anaconda/bin/activate py37
python3 -m pip install "$hail_pip_specifier" --no-dependencies
site_packages=$(python3 -m pip show hail| grep -E 'Location:' | sed -E 's/^Location: //')
# https://www.python.org/dev/peps/pep-0440/#version-specifiers
grep -E '^Requires-Dist:' $site_packages/hail-$hail_pip_version.dist-info/METADATA \
    | sed 's/Requires-Dist: //' \
    | sed -E 's/ \(([^)]*)\)/\1/' \
    | grep -Ev '^pyspark(~=|==|!=|<=|>=|<|>|===|$)' \
    > requirements.txt
python3 -m pip install -r requirements.txt
/usr/bin/anaconda/bin/conda install ipykernel --yes
python3 -m ipykernel install

spark_monitor_gs=https://storage.googleapis.com/hail-common/sparkmonitor-3357488112c6c162c12f8386faaadcbf3789ac02/sparkmonitor-0.0.12-py3-none-any.whl
curl -LO $spark_monitor_gs
python3 -m pip install $(basename $spark_monitor_gs) widgetsnbextension
jupyter serverextension enable --user --py sparkmonitor
jupyter nbextension install --user --py sparkmonitor
jupyter nbextension enable --user --py sparkmonitor
jupyter nbextension enable --user --py widgetsnbextension
mkdir -p /home/spark/.ipython/profile_default
echo "c.InteractiveShellApp.extensions.append('sparkmonitor.kernelextension')" \
     >> /home/spark/.ipython/profile_default/ipython_kernel_config.py

chown -R spark:spark /home/spark

mv /usr/bin/anaconda/envs/py37/bin/python /usr/bin/anaconda/envs/py37/bin/python.bak
