echo 'umask 022' >> ~/.profile
echo 'export HAIL=$HOME/hail' >> ~/.profile
echo 'source $HAIL/devbin/functions.sh' >> ~/.profile
umask 022

sudo apt update
sudo snap install --classic astral-uv
uv venv --seed --python 3.12
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
sudo apt update

sudo apt install -y build-essential docker.io python3-pip openjdk-11-jdk-headless jq skopeo docker-buildx-plugin
sudo snap install --classic kubectl
sudo snap install --classic aws-cli
sudo usermod -a -G docker $USER

sudo snap install --classic helm

python3 -m pip install --upgrade pip
make -C $HOME/hail install-dev-requirements
