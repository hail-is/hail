#!/bin/bash
# Install a Hail wheel on every EMR node.
# Arguments:
#   1 wheel S3 URI

set -euxo pipefail

wheel_uri=$1
wheel_filename=${wheel_uri##*/}
if [[ ! "${wheel_filename}" =~ ^hail-.+\.whl$ ]]; then
    echo "wheel URI does not end in a valid Hail wheel filename: ${wheel_uri}" >&2
    exit 1
fi
wheel_path=/tmp/${wheel_filename}

sudo dnf install -y gcc-c++ openblas-devel lapack-devel python3.12 python3.12-pip

aws s3 cp "${wheel_uri}" "${wheel_path}"
sudo python3.12 -m pip install "${wheel_path}"
# EMR provides Spark; a second PySpark from PyPI would shadow it.
sudo python3.12 -m pip uninstall -y pyspark

site_packages=$(sudo python3.12 -m pip show hail | grep -E '^Location:' | sed -E 's/^Location: //')
if [[ -z "${site_packages}" ]]; then
    echo "could not determine where pip installed hail" >&2
    exit 1
fi

sudo mkdir -p /usr/lib/hail
sudo cp "${site_packages}/hail/backend/hail-all-spark.jar" /usr/lib/hail/hail-all-spark.jar

echo 'HAIL_CLOUD=aws' | sudo tee -a /etc/environment
