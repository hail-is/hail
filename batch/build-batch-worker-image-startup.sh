#!/bin/bash

set -ex

curl --silent --show-error --remote-name --fail https://dl.google.com/cloudagents/add-logging-agent-repo.sh
bash add-logging-agent-repo.sh

apt-get update

apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    google-fluentd \
    google-fluentd-catch-all-config-structured \
    jq \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

apt-get install -y docker-ce

rm -rf /var/lib/apt/lists/*

[ -f /etc/docker/daemon.json ] || echo "{}" > /etc/docker/daemon.json

VERSION=2.0.4
OS=linux
ARCH=amd64

curl -fsSL "https://github.com/GoogleCloudPlatform/docker-credential-gcr/releases/download/v${VERSION}/docker-credential-gcr_${OS}_${ARCH}-${VERSION}.tar.gz" \
  | tar xz --to-stdout ./docker-credential-gcr \
	> /usr/bin/docker-credential-gcr && chmod +x /usr/bin/docker-credential-gcr

# avoid "unable to get current user home directory: os/user lookup failed"
export HOME=/root

DOCKER_PREFIX=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/docker_prefix")

docker-credential-gcr configure-docker --include-artifact-registry

docker pull ${DOCKER_PREFIX}/ubuntu:18.04
docker pull gcr.io/google.com/cloudsdktool/cloud-sdk:310.0.0-alpine

# add docker daemon debug logging
jq '.debug = true' /etc/docker/daemon.json > daemon.json.tmp
mv daemon.json.tmp /etc/docker/daemon.json

shutdown -h now
