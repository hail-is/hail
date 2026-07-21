#!/bin/bash
# EMR bootstrap action: install Hail from PyPI onto every node.
# Bootstrap actions run BEFORE Spark is installed, so we only install the
# Python package here and place the bundled JAR at a deterministic path that
# the cluster's --configurations reference (see start.py).
#
# Argument 1: the Hail pip version to install (e.g. 0.2.140).

set -ex

hail_pip_version=$1

sudo dnf install -y gcc-c++ openblas-devel lapack-devel

# Install Hail itself without its dependencies, so we can drop pyspark
# (EMR provides pyspark) before installing the rest.
sudo python3 -m pip install "hail==${hail_pip_version}" --no-dependencies

site_packages=$(python3 -m pip show hail | grep -E '^Location:' | sed -E 's/^Location: //')

# Reinstall Hail's declared dependencies, excluding pyspark.
# See https://www.python.org/dev/peps/pep-0440/#version-specifiers
grep -E '^Requires-Dist:' "${site_packages}/hail-${hail_pip_version}.dist-info/METADATA" \
    | sed 's/Requires-Dist: //' \
    | sed -E 's/ \(([^)]*)\)/\1/' \
    | grep -Ev '^pyspark(~=|==|!=|<=|>=|<|>|===|$)' \
    > /tmp/hail-requirements.txt
sudo python3 -m pip install -r /tmp/hail-requirements.txt

# Copy the bundled JAR to a fixed path referenced by spark-defaults.
sudo mkdir -p /usr/lib/hail
sudo cp "${site_packages}/hail/backend/hail-all-spark.jar" /usr/lib/hail/hail-all-spark.jar
