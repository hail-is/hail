#!/bin/bash

set -ex

vep_loftee_location=$1
vep_homo_sapiens_location=$2
assembly=$3
vep_docker_image=$4
vep_config_uri=$5

mkdir -p /vep_data/loftee_data
mkdir -p /vep_data/homo_sapiens

apt-get update
apt-get -y install \
    ca-certificates \
    curl \
    software-properties-common \
    tabix

setup_docker () {
    apt-get update
    apt-get install -y docker.io
    docker pull $vep_docker_image
}

mkdir -p /vep_data/homo_sapiens

(
    set -ex
    time setup_docker
    echo $? > docker-ec
) 1>docker-out 2>&1 &
(
    set -ex
    time curl -fsSL $vep_config_uri > /vep_data/vep-azure.json
    echo $? > config-ec
) 1>config-out 2>&1 &
(
    set -ex
    time hadoop fs -copyToLocal $vep_loftee_location'/*' /vep_data/
    echo $? > loftee-ec
) 1>loftee-out 2>&1 &
(
    set -ex
    time hadoop fs -copyToLocal $vep_homo_sapiens_location'/*' /vep_data/homo_sapiens/
    echo $? > homo-sapiens-ec
) 1>homo-sapiens-out 2>&1 &

wait

exit_code=0
for i in docker config loftee homo-sapiens
do
    if [ $(cat $i-ec) -ne 0 ]
    then
        echo ">>> $i FAILED <<<"
        exit_code=1
    else
        echo ">>> $i success <<<"
    fi
    cat $i-out
done

if [ $exit_code -ne 0 ]
then
    exit $exit_code
fi

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

if [ $assembly == "GRCh37" ]
then
    cat >/vep.sh <<EOF
#!/bin/bash

docker run -i -v /vep_data:/root/.vep:ro ${vep_docker_image} \
  perl /vep/ensembl-tools-release-85/scripts/variant_effect_predictor/variant_effect_predictor.pl \
  "\$@"
EOF
    chmod +x /vep.sh

    # Run VEP on the 1-variant VCF to create fasta.index file -- caution do not make fasta.index file writeable afterwards!
    cat /vep_data/loftee_data/1var.vcf | docker run -i -v /vep_data:/root/.vep \
        ${vep_docker_image} \
        perl /vep/ensembl-tools-release-85/scripts/variant_effect_predictor/variant_effect_predictor.pl \
        --format vcf \
        --json \
        --everything \
        --allele_number \
        --no_stats \
        --cache --offline \
        --minimal \
        --assembly ${assembly} \
        -o STDOUT
elif [ $assembly == "GRCh38" ]
then
    cat >/vep.sh <<EOF
#!/bin/bash

docker run -i -v /vep_data/:/opt/vep/.vep/:ro ${vep_docker_image} \
  /opt/vep/src/ensembl-vep/vep "\$@"
EOF
    chmod +x /vep.sh
else
    echo "Unknown assembly: $assembly"
    exit 1
fi
