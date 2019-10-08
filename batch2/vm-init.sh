#! /bin/bash

sudo apt-get update

sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common && \
    sudo rm -rf /var/lib/apt/lists/*

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get install -y docker-ce && \
    sudo rm -rf /var/lib/apt/lists/*

VERSION=1.5.0
OS=linux 
ARCH=amd64

curl -fsSL "https://github.com/GoogleCloudPlatform/docker-credential-gcr/releases/download/v${VERSION}/docker-credential-gcr_${OS}_${ARCH}-${VERSION}.tar.gz" \
  | sudo tar xz --to-stdout ./docker-credential-gcr \
	> /usr/bin/docker-credential-gcr && sudo chmod +x /usr/bin/docker-credential-gcr

sudo docker-credential-gcr configure-docker

sudo docker pull ubuntu:18.04
sudo docker pull google/cloud-sdk:237.0.0-alpine
sudo docker pull gcr.io/hail-vdc/batch2:latest
