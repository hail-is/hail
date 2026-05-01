#!/bin/bash

set -exo pipefail

# The nvidia toolkit must be installed in this startup script to be able to configure docker with the command nvidia-ctk runtime configure --runtime=docker. This command cannot be run from Dockerfile.worker
# The toolkit also has to be installed in Dockerfile.worker since to execute the nvidia hook, the toolkit needs to be installed in the container from which crun is invoked.

echo "=== Installing base packages ==="
apt-get update

apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    jq \
    software-properties-common

echo "=== Adding Docker apt repo ==="
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

echo "=== Installing Docker ==="
apt-get install -y docker-ce

rm -rf /var/lib/apt/lists/*

[ -f /etc/docker/daemon.json ] || echo "{}" > /etc/docker/daemon.json

echo "=== Installing docker-credential-gcr ==="
VERSION=2.0.4
OS=linux
ARCH=amd64

curl -fsSL "https://github.com/GoogleCloudPlatform/docker-credential-gcr/releases/download/v${VERSION}/docker-credential-gcr_${OS}_${ARCH}-${VERSION}.tar.gz" \
  | tar xz --to-stdout ./docker-credential-gcr \
	> /usr/bin/docker-credential-gcr && chmod +x /usr/bin/docker-credential-gcr

echo "=== Adding NVIDIA container toolkit apt repo ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
     | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "=== Installing build tools and kernel headers ==="
apt-get update
apt-get install -y build-essential linux-headers-$(uname -r)
apt-get install -y gcc-12
apt-get install -y g++-12

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 50
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 50

echo "=== Downloading NVIDIA driver ==="
wget --no-verbose https://us.download.nvidia.com/XFree86/Linux-x86_64/535.183.01/NVIDIA-Linux-x86_64-535.183.01.run
chmod +x NVIDIA-Linux-x86_64-535.183.01.run

echo "=== Running NVIDIA driver installer ==="
touch /var/log/nvidia-installer.log
tail -f /var/log/nvidia-installer.log &
NVIDIA_LOG_PID=$!
./NVIDIA-Linux-x86_64-535.183.01.run --silent
kill $NVIDIA_LOG_PID

echo "=== Installing NVIDIA container toolkit ==="
apt-get --yes install nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker

echo "=== Installing Google Cloud Ops Agent ==="
curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
bash add-google-cloud-ops-agent-repo.sh --also-install --version=2.*.*

echo "=== Configuring Docker with GCR credentials and pulling root image ==="
# avoid "unable to get current user home directory: os/user lookup failed"
export HOME=/root

docker-credential-gcr configure-docker --include-artifact-registry
docker pull {{ global.docker_root_image }}

echo "=== Enabling Docker debug logging ==="
jq '.debug = true' /etc/docker/daemon.json > daemon.json.tmp
mv daemon.json.tmp /etc/docker/daemon.json

systemctl restart docker

echo "=== Startup complete, shutting down ==="
shutdown -h now
