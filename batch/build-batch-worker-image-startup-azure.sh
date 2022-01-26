#!/bin/bash

set -ex

apt-get update

apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    jq \
    lsb-release \
    software-properties-common

# Install Docker

curl --connect-timeout 5 \
     --max-time 10 \
     --retry 5 \
     --retry-max-time 40 \
     --location \
     --fail \
     --silent \
     --show-error \
     https://download.docker.com/linux/ubuntu/gpg | apt-key add -

add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

apt-get install -y docker-ce

rm -rf /var/lib/apt/lists/*

[ -f /etc/docker/daemon.json ] || echo "{}" > /etc/docker/daemon.json

# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

az login --identity
az acr login --name {{ global.container_registry_name }}

# avoid "unable to get current user home directory: os/user lookup failed"
export HOME=/root

docker pull {{ global.docker_root_image }}

# add docker daemon debug logging
jq '.debug = true' /etc/docker/daemon.json > daemon.json.tmp
mv daemon.json.tmp /etc/docker/daemon.json
