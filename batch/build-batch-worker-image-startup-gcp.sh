#!/bin/bash

set -ex

# Install required packages for this script (build-batch-worker-image-startup-gcp.sh)
apt-get update
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    jq

# add-logging-agent-repo.sh script adds the google repositories and latest apt key
curl --silent --show-error --remote-name --fail https://dl.google.com/cloudagents/add-logging-agent-repo.sh
bash add-logging-agent-repo.sh

# Add other needed repositories: docker, nvidia
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
     | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/ubuntu24.04/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y \
    build-essential \
    docker-ce \
    g++-12 \
    gcc-12 \
    google-fluentd \
    google-fluentd-catch-all-config-structured \
    software-properties-common

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 50
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 50

[ -f /etc/docker/daemon.json ] || echo "{}" > /etc/docker/daemon.json

VERSION=2.1.30
OS=linux
ARCH=amd64

curl -fsSL "https://github.com/GoogleCloudPlatform/docker-credential-gcr/releases/download/v${VERSION}/docker-credential-gcr_${OS}_${ARCH}-${VERSION}.tar.gz" \
  | tar xz --to-stdout ./docker-credential-gcr \
	> /usr/bin/docker-credential-gcr && chmod +x /usr/bin/docker-credential-gcr

# The nvidia toolkit must be installed in this startup script to be able to configure docker with the command nvidia-ctk runtime configure --runtime=docker. This command cannot be run from Dockerfile.worker
# The toolkit also has to be installed in Dockerfile.worker since to execute the nvidia hook, the toolkit needs to be installed in the container from which crun is invoked.
wget --no-verbose https://us.download.nvidia.com/XFree86/Linux-x86_64/535.183.01/NVIDIA-Linux-x86_64-535.183.01.run
chmod +x NVIDIA-Linux-x86_64-535.183.01.run
./NVIDIA-Linux-x86_64-535.183.01.run --silent

apt-get --yes install nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker

curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
bash add-google-cloud-ops-agent-repo.sh --also-install --version=2.*.*

# avoid "unable to get current user home directory: os/user lookup failed"
export HOME=/root

docker-credential-gcr configure-docker --include-artifact-registry
docker pull {{ global.docker_root_image }}

# add docker daemon debug logging
jq '.debug = true' /etc/docker/daemon.json > daemon.json.tmp
mv daemon.json.tmp /etc/docker/daemon.json

systemctl restart docker

rm -rf /var/lib/apt/lists/*

shutdown -h now
