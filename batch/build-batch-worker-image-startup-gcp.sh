#!/bin/bash

set -ex

curl --silent --show-error --remote-name --fail https://dl.google.com/cloudagents/add-logging-agent-repo.sh
bash add-logging-agent-repo.sh


# Get the latest GPG key as it might not always be up to date
# https://cloud.google.com/compute/docs/troubleshooting/known-issues#keyexpired
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
source /etc/os-release
sh -c "echo 'deb http://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_${VERSION_ID}/ /' > /etc/apt/sources.list.d/devel:kubic:libcontainers:stable.list"
wget -nv https://download.opensuse.org/repositories/devel:kubic:libcontainers:stable/xUbuntu_${VERSION_ID}/Release.key -O- | sudo apt-key add -
apt-get update

apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    google-fluentd \
    google-fluentd-catch-all-config-structured \
    jq \
    software-properties-common \
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
