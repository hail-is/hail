echo 'umask 022' >> ~/.profile
echo 'export HAIL=$HOME/hail' >> ~/.profile
echo 'source $HAIL/devbin/functions.sh' >> ~/.profile
umask 022

# Necessary to install Skopeo on 20.04 (can be removed on 20.10)
. /etc/os-release
echo "deb https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_${VERSION_ID}/ /" | sudo tee /etc/apt/sources.list.d/devel:kubic:libcontainers:stable.list
curl -L https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_${VERSION_ID}/Release.key | sudo apt-key add -

sudo apt update
sudo apt install -y docker.io python3-pip openjdk-8-jdk-headless jq skopeo
sudo snap install --classic kubectl
sudo usermod -a -G docker $USER

python3 -m pip install --upgrade pip
make -C $HOME/hail install-dev-requirements
