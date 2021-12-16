#!/bin/bash

set -ex

hail_pip_version=$1

apt-get install -y \
    g++ \
    libopenblas-base liblapack3

/usr/bin/anaconda/bin/conda create -n py37 python=3.7 --yes
source /usr/bin/anaconda/bin/activate py37
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
/usr/bin/anaconda/bin/conda install ipykernel --yes
python3 -m ipykernel install --user
rm -rf /usr/bin/anaconda/envs/py37/bin/python

# curl -u admin:LongPassword1 -H 'X-Requested-By: ambari' -X PUT -d '{
#     "RequestInfo": {"context": "put services into STOPPED state"},
#     "Body": {"ServiceInfo": {"state" : "INSTALLED"}}
# }' https://dkingtest25.azurehdinsight.net/api/v1/clusters/dkingtest25/services/JUPYTER/

# sleep 15

# curl -u admin:LongPassword1 -H 'X-Requested-By: ambari' -X PUT -d '{
#     "RequestInfo": {"context": "put services into STARTED state"},
#     "Body": {"ServiceInfo": {"state" : "STARTED"}}
# }' https://dkingtest25.azurehdinsight.net/api/v1/clusters/dkingtest25/services/JUPYTER/
