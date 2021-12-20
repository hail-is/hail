echo 'umask 022' >> ~/.profile
echo 'source ~/hail/devbin/functions.sh' >> ~/.profile
umask 022
sudo apt update
sudo apt install -y docker.io python3-pip openjdk-8-jre-headless jq
sudo snap install --classic kubectl
sudo usermod -a -G docker $USER

# Terraform
curl --connect-timeout 5 \
     --max-time 10 \
     --retry 5 \
     --retry-all-errors \
     --retry-max-time 40 \
     --location \
     --fail \
     --silent \
     --show-error \
     https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
sudo apt-get update && sudo apt-get install terraform

python3 -m pip install --upgrade pip
python3 -m pip install -r $HOME/hail/docker/requirements.txt
