echo 'umask 022' >> ~/.profile
umask 022
sudo apt update
sudo apt install -y docker.io python3-pip openjdk-8-jdk-headless jq
sudo snap install --classic kubectl
sudo usermod -a -G docker $USER

# Terraform
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
sudo apt-get update && sudo apt-get install terraform

python3 -m pip install -r $HOME/hail/docker/requirements.txt
