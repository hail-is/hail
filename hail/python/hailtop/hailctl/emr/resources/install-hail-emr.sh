#!/bin/bash
# Install an exact, checksum-verified Hail Spark 4 artifact on every EMR node.
# Arguments:
#   1 wheel S3 URI
#   2 wheel SHA256
#   3 wheelhouse tar.gz S3 URI
#   4 wheelhouse SHA256

set -euxo pipefail

wheel_uri=$1
wheel_sha256=$2
wheelhouse_uri=$3
wheelhouse_sha256=$4
workdir=/tmp/hail-emr-bootstrap
wheel_path=${workdir}/hail.whl
wheelhouse_archive=${workdir}/wheelhouse.tar.gz
wheelhouse_dir=${workdir}/wheelhouse

mkdir -p "${workdir}" "${wheelhouse_dir}"

sudo dnf install -y gcc-c++ openblas-devel lapack-devel python3.12 python3.12-pip

aws s3 cp "${wheel_uri}" "${wheel_path}"
echo "${wheel_sha256}  ${wheel_path}" | sha256sum -c -
aws s3 cp "${wheelhouse_uri}" "${wheelhouse_archive}"
echo "${wheelhouse_sha256}  ${wheelhouse_archive}" | sha256sum -c -
tar -xzf "${wheelhouse_archive}" -C "${wheelhouse_dir}"

requirements=${wheelhouse_dir}/requirements.txt
if [[ ! -f "${requirements}" ]]; then
    echo "wheelhouse does not contain requirements.txt" >&2
    exit 1
fi

sudo python3.12 -m pip install "${wheel_path}" --no-dependencies
sudo python3.12 -m pip install --no-index --find-links "${wheelhouse_dir}" -r "${requirements}"

site_packages=$(sudo python3.12 -m pip show hail | grep -E '^Location:' | sed -E 's/^Location: //')
if [[ -z "${site_packages}" ]]; then
    echo "could not determine where pip installed hail" >&2
    exit 1
fi

sudo mkdir -p /usr/lib/hail
sudo cp "${site_packages}/hail/backend/hail-all-spark.jar" /usr/lib/hail/hail-all-spark.jar

echo 'HAIL_CLOUD=aws' | sudo tee -a /etc/environment
