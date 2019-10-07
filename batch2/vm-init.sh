#! /bin/bash

sudo apt-get update

sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common && \
    rm -rf /var/lib/apt/lists/*

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get install -y docker-ce && \
    rm -rf /var/lib/apt/lists/*

VERSION=1.5.0
OS=linux 
ARCH=amd64

curl -fsSL "https://github.com/GoogleCloudPlatform/docker-credential-gcr/releases/download/v${VERSION}/docker-credential-gcr_${OS}_${ARCH}-${VERSION}.tar.gz" \
  | tar xz --to-stdout ./docker-credential-gcr \
	> /usr/bin/docker-credential-gcr && chmod +x /usr/bin/docker-credential-gcr

docker-credential-gcr configure-docker
