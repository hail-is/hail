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

VERSION=1.5.0
OS=linux 
ARCH=amd64

curl -fsSL "https://github.com/GoogleCloudPlatform/docker-credential-gcr/releases/download/v${VERSION}/docker-credential-gcr_${OS}_${ARCH}-${VERSION}.tar.gz" \
  | tar xz --to-stdout ./docker-credential-gcr \
	> /usr/bin/docker-credential-gcr && chmod +x /usr/bin/docker-credential-gcr

docker-credential-gcr configure-docker

docker pull ubuntu:18.04
docker pull google/cloud-sdk:269.0.0-alpine

service google-fluentd start

shutdown -h now
