#!/bin/bash

export ASSEMBLY=GRCh38
export VEP_DOCKER_IMAGE=konradjk/vep95_loftee:0.2

mkdir -p /vep_data/loftee_data
mkdir -p /vep_data/homo_sapiens

# Install docker
apt-get update
apt-get -y install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg2 \
    software-properties-common \
    tabix
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"
apt-get update
apt-get install -y --allow-unauthenticated docker-ce

# Get VEP cache and LOFTEE data
gsutil cp gs://hail-common/vep/vep/vep95-GRCh38-loftee-gcloud.json /vep_data/vep95-GRCh38-gcloud.json
ln -s /vep_data/vep95-GRCh38-gcloud.json /vep_data/vep-gcloud.json

gsutil -m cp -r gs://hail-common/vep/vep/loftee-beta/${ASSEMBLY}/* /vep_data/ &
gsutil -m cp -r gs://hail-common/vep/vep/Plugins /vep_data &
gsutil -m cp -r gs://hail-common/vep/vep/homo_sapiens/95_${ASSEMBLY} /vep_data/homo_sapiens/ &
docker pull ${VEP_DOCKER_IMAGE} &
wait

cat >/vep.c <<EOF
#include <unistd.h>
#include <stdio.h>

int
main(int argc, char *const argv[]) {
  if (setuid(geteuid()))
    perror( "setuid" );

  execv("/vep.sh", argv);
  return 0;
}
EOF
gcc -Wall -Werror -O2 /vep.c -o /vep
chmod u+s /vep

cat >/vep.sh <<EOF
#!/bin/bash

docker run -i -v /vep_data/:/opt/vep/.vep/:ro ${VEP_DOCKER_IMAGE} \
  /opt/vep/src/ensembl-vep/vep "\$@"
EOF
chmod +x /vep.sh
