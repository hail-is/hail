echo 'umask 022' >> ~/.profile
echo 'export HAIL=$HOME/hail' >> ~/.profile
echo 'source $HAIL/devbin/functions.sh' >> ~/.profile
umask 022

sudo apt update
sudo apt install -y docker.io python3-pip openjdk-8-jre-headless jq
sudo snap install --classic kubectl
sudo usermod -a -G docker $USER

python3 -m pip install --upgrade pip
python3 -m pip install -r $HOME/hail/docker/requirements.txt
