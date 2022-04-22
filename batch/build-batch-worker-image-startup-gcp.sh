#!/bin/bash

set -ex

curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
sudo bash add-google-cloud-ops-agent-repo.sh --also-install

# Get the latest GPG key as it might not always be up to date
# https://cloud.google.com/compute/docs/troubleshooting/known-issues#keyexpired
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
source /etc/os-release
echo 'deb http://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/Debian_11/ /' | sudo tee /etc/apt/sources.list.d/devel:kubic:libcontainers:stable.list
curl -fsSL https://download.opensuse.org/repositories/devel:kubic:libcontainers:stable/Debian_11/Release.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/devel_kubic_libcontainers_stable.gpg > /dev/null
apt-get update

apt-get install -y \
    apt-transport-https \
    ca-certificates \
    jq \
    software-properties-common \
    xfsprogs
    podman

rm -rf /var/lib/apt/lists/*

VERSION=2.0.4
OS=linux
ARCH=amd64

# avoid "unable to get current user home directory: os/user lookup failed"
export HOME=/root

gcloud auth print-access-token | podman login -u oauth2accesstoken --password-stdin {{ global.docker_prefix }}
podman pull {{ global.docker_root_image }}

shutdown -h now
