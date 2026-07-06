echo 'umask 022' >> ~/.profile
echo 'export HAIL=$HOME/hail' >> ~/.profile
echo 'source $HAIL/devbin/functions.sh' >> ~/.profile
umask 022

# Necessary to install Skopeo on 20.04 (can be removed on 20.10)
. /etc/os-release
#echo "deb https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_${VERSION_ID}/ /" | sudo tee /etc/apt/sources.list.d/devel:kubic:libcontainers:stable.list
#curl -L https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_${VERSION_ID}/Release.key | sudo apt-key add -

sudo apt update
sudo snap install --classic astral-uv
uv venv --seed --python 3.11
source .venv/bin/activate

# Add Docker's official GPG key:
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt install -y build-essential docker.io python3-pip openjdk-11-jdk-headless jq skopeo docker-buildx-plugin
sudo snap install --classic kubectl
sudo snap install --classic aws-cli
sudo usermod -a -G docker $USER
gcloud components install gke-gcloud-auth-plugin


python3 -m pip install --upgrade pip
make -C $HOME/hail install-dev-requirements
