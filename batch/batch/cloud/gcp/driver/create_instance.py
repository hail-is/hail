import base64
import json
import logging
import os
from shlex import quote as shq
from typing import Dict, List

from gear.cloud_config import get_global_config
from hailtop.config import get_deploy_config

from ....batch_configuration import DEFAULT_NAMESPACE, DOCKER_PREFIX, DOCKER_ROOT_IMAGE, INTERNAL_GATEWAY_IP
from ....file_store import FileStore
from ....instance_config import InstanceConfig
from ...resource_utils import unreserved_worker_data_disk_size_gib
from ...utils import ACCEPTABLE_QUERY_JAR_URL_PREFIX
from ..resource_utils import gcp_machine_type_to_worker_type_and_cores

log = logging.getLogger('create_instance')

BATCH_WORKER_IMAGE = os.environ['HAIL_BATCH_WORKER_IMAGE']


log.info(f'BATCH_WORKER_IMAGE {BATCH_WORKER_IMAGE}')
log.info(f'ACCEPTABLE_QUERY_JAR_URL_PREFIX {ACCEPTABLE_QUERY_JAR_URL_PREFIX}')


def create_vm_config(
    file_store: FileStore,
    resource_rates: Dict[str, float],
    zone: str,
    machine_name: str,
    machine_type: str,
    activation_token: str,
    max_idle_time_msecs: int,
    local_ssd_data_disk: bool,
    data_disk_size_gb: int,
    boot_disk_size_gb: int,
    preemptible: bool,
    job_private: bool,
    project: str,
    instance_config: InstanceConfig,
) -> dict:
    _, cores = gcp_machine_type_to_worker_type_and_cores(machine_type)

    region = instance_config.region_for(zone)

    if local_ssd_data_disk:
        worker_data_disk = {
            'type': 'SCRATCH',
            'autoDelete': True,
            'interface': 'NVME',
            'initializeParams': {'diskType': f'zones/{zone}/diskTypes/local-ssd'},
        }
        worker_data_disk_name = 'nvme0n1'
    else:
        worker_data_disk = {
            'autoDelete': True,
            'initializeParams': {
                'diskType': f'projects/{project}/zones/{zone}/diskTypes/pd-ssd',
                'diskSizeGb': str(data_disk_size_gb),
            },
        }
        worker_data_disk_name = 'sdb'

    if job_private:
        unreserved_disk_storage_gb = data_disk_size_gb
    else:
        unreserved_disk_storage_gb = unreserved_worker_data_disk_size_gib(data_disk_size_gb, cores)
    assert unreserved_disk_storage_gb >= 0

    make_global_config = ['mkdir /global-config']
    global_config = get_global_config()
    for name, value in global_config.items():
        make_global_config.append(f'echo -n {shq(value)} > /global-config/{name}')
    make_global_config_str = '\n'.join(make_global_config)

    assert instance_config.is_valid_configuration(resource_rates.keys())

    configs: List[str] = []
    touch_commands = []
    for jvm_cores in (1, 2, 4, 8):
        for _ in range(cores // jvm_cores):
            idx = len(configs)
            log_path = f'/batch/jvm-container-logs/jvm-{idx}.log'
            touch_commands.append(f'touch {log_path}')

            config = f'''
<source>
@type tail
<parse>
    # 'none' indicates the log is unstructured (text).
    @type none
</parse>
path {log_path}
pos_file /var/lib/google-fluentd/pos/jvm-{idx}.pos
read_from_head true
tag jvm-{idx}.log
</source>
'''
            configs.append(config)

    jvm_fluentd_config = '\n'.join(configs)
    jvm_touch_command = '\n'.join(touch_commands)

    def scheduling() -> dict:
        result = {
            'automaticRestart': False,
            'onHostMaintenance': 'TERMINATE',
            'preemptible': preemptible,
        }

        if preemptible:
            result.update(
                {
                    'provisioningModel': 'SPOT',
                    'instanceTerminationAction': 'DELETE',
                }
            )

        return result

    return {
        'name': machine_name,
        'machineType': f'projects/{project}/zones/{zone}/machineTypes/{machine_type}',
        'labels': {'role': 'batch2-agent', 'namespace': DEFAULT_NAMESPACE},
        'disks': [
            {
                'boot': True,
                'autoDelete': True,
                'initializeParams': {
                    'sourceImage': f'projects/{project}/global/images/batch-worker-12',
                    'diskType': f'projects/{project}/zones/{zone}/diskTypes/pd-ssd',
                    'diskSizeGb': str(boot_disk_size_gb),
                },
            },
            worker_data_disk,
        ],
        'networkInterfaces': [
            {
                'network': 'global/networks/default',
                'subnetwork': f'regions/{region}/subnetworks/default',
                'networkTier': 'PREMIUM',
                'accessConfigs': [{'type': 'ONE_TO_ONE_NAT', 'name': 'external-nat'}],
            }
        ],
        'scheduling': scheduling(),
        'serviceAccounts': [
            {
                'email': f'batch2-agent@{project}.iam.gserviceaccount.com',
                'scopes': ['https://www.googleapis.com/auth/cloud-platform'],
            }
        ],
        'metadata': {
            'items': [
                {
                    'key': 'startup-script',
                    'value': '''
#!/bin/bash
set -x

NAME=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google')
ZONE=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google')

if [ -f "/started" ]; then
    echo "instance $NAME has previously been started"
    while true; do
    gcloud -q compute instances delete $NAME --zone=$ZONE
    sleep 1
    done
    exit
else
    touch /started
fi

curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/run_script"  >./run.sh

nohup /bin/bash run.sh >run.log 2>&1 &
    ''',
                },
                {
                    'key': 'run_script',
                    'value': rf'''
#!/bin/bash
set -x

WORKER_DATA_DISK_NAME="{worker_data_disk_name}"
UNRESERVED_WORKER_DATA_DISK_SIZE_GB="{unreserved_disk_storage_gb}"
ACCEPTABLE_QUERY_JAR_URL_PREFIX="{ACCEPTABLE_QUERY_JAR_URL_PREFIX}"

# format worker data disk
sudo mkfs.xfs -m reflink=1 -n ftype=1 /dev/$WORKER_DATA_DISK_NAME
sudo mkdir -p /mnt/disks/$WORKER_DATA_DISK_NAME
sudo mount -o prjquota /dev/$WORKER_DATA_DISK_NAME /mnt/disks/$WORKER_DATA_DISK_NAME
sudo chmod a+w /mnt/disks/$WORKER_DATA_DISK_NAME
XFS_DEVICE=$(xfs_info /mnt/disks/$WORKER_DATA_DISK_NAME | head -n 1 | awk '{{ print $1 }}' | awk  'BEGIN {{ FS = "=" }}; {{ print $2 }}')

# reconfigure docker to use local SSD
sudo service docker stop
sudo mv /var/lib/docker /mnt/disks/$WORKER_DATA_DISK_NAME/docker
sudo ln -s /mnt/disks/$WORKER_DATA_DISK_NAME/docker /var/lib/docker
sudo service docker start

# reconfigure /batch and /logs and /gcsfuse to use local SSD
sudo mkdir -p /mnt/disks/$WORKER_DATA_DISK_NAME/batch/
sudo ln -s /mnt/disks/$WORKER_DATA_DISK_NAME/batch /batch

sudo mkdir -p /mnt/disks/$WORKER_DATA_DISK_NAME/logs/
sudo ln -s /mnt/disks/$WORKER_DATA_DISK_NAME/logs /logs

sudo mkdir -p /mnt/disks/$WORKER_DATA_DISK_NAME/cloudfuse/
sudo ln -s /mnt/disks/$WORKER_DATA_DISK_NAME/cloudfuse /cloudfuse

sudo mkdir -p /etc/netns

# Setup fluentd
touch /worker.log
touch /run.log

sudo rm /etc/google-fluentd/config.d/*  # remove unused config files

sudo tee /etc/google-fluentd/config.d/worker-log.conf <<EOF
<source>
@type tail
format json
path /worker.log
pos_file /var/lib/google-fluentd/pos/worker-log.pos
read_from_head true
tag worker.log
</source>

<filter worker.log>
@type record_transformer
enable_ruby
<record>
    severity \${{ record["levelname"] }}
    timestamp \${{ record["asctime"] }}
</record>
</filter>
EOF

sudo tee /etc/google-fluentd/config.d/syslog.conf <<EOF
<source>
@type tail
format syslog
path /var/log/syslog
pos_file /var/lib/google-fluentd/pos/syslog.pos
read_from_head true
tag syslog
</source>
EOF

sudo tee /etc/google-fluentd/config.d/run-log.conf <<EOF
<source>
@type tail
format none
path /run.log
pos_file /var/lib/google-fluentd/pos/run-log.pos
read_from_head true
tag run.log
</source>
EOF

sudo tee /etc/google-fluentd/config.d/jvm-logs.conf <<EOF
{jvm_fluentd_config}
EOF

sudo cp /etc/google-fluentd/google-fluentd.conf /etc/google-fluentd/google-fluentd.conf.bak
head -n -1 /etc/google-fluentd/google-fluentd.conf.bak | sudo tee /etc/google-fluentd/google-fluentd.conf
sudo tee -a /etc/google-fluentd/google-fluentd.conf <<EOF
labels {{
"namespace": "$NAMESPACE",
"instance_id": "$INSTANCE_ID"
}}
</match>
EOF
rm /etc/google-fluentd/google-fluentd.conf.bak

mkdir -p /batch/jvm-container-logs/
{jvm_touch_command}

sudo service google-fluentd restart

CORES=$(nproc)
NAMESPACE=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/namespace")
ACTIVATION_TOKEN=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/activation_token")
IP_ADDRESS=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip")
PROJECT=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/project/project-id")

BATCH_LOGS_STORAGE_URI=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/batch_logs_storage_uri")
INSTANCE_ID=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/instance_id")
INSTANCE_CONFIG=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/instance_config")
MAX_IDLE_TIME_MSECS=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/max_idle_time_msecs")
REGION=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/region")

NAME=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google')
ZONE=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google')

BATCH_WORKER_IMAGE=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/batch_worker_image")
DOCKER_ROOT_IMAGE=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/docker_root_image")
DOCKER_PREFIX=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/docker_prefix")

INTERNAL_GATEWAY_IP=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/internal_ip")

# private job network = 172.20.0.0/16
# public job network = 172.21.0.0/16
# [all networks] Rewrite traffic coming from containers to masquerade as the host
iptables --table nat --append POSTROUTING --source 172.20.0.0/15 --jump MASQUERADE

# [public]
# Block public traffic to the metadata server
iptables --append FORWARD --source 172.21.0.0/16 --destination 169.254.169.254 --jump DROP
# But allow the internal gateway
iptables --append FORWARD --destination $INTERNAL_GATEWAY_IP --jump ACCEPT
# And this worker
iptables --append FORWARD --destination $IP_ADDRESS --jump ACCEPT
# Allow traffic going to the internet
INTERNET_INTERFACE=$(ip link list | grep ens | awk -F": " '{{ print $2 }}')
iptables --append FORWARD --out-interface $INTERNET_INTERFACE ! --destination 10.128.0.0/16 --jump ACCEPT

# [private]
# Allow all traffic from the private job network
iptables --append FORWARD --source 172.20.0.0/16 --jump ACCEPT

{make_global_config_str}

mkdir /deploy-config
cat >/deploy-config/deploy-config.json <<EOF
{ json.dumps(get_deploy_config().with_location('gce').get_config()) }
EOF


# retry once
docker pull $BATCH_WORKER_IMAGE || \
(echo 'pull failed, retrying' && sleep 15 && docker pull $BATCH_WORKER_IMAGE)

BATCH_WORKER_IMAGE_ID=$(docker inspect $BATCH_WORKER_IMAGE --format='{{{{.Id}}}}' | cut -d':' -f2)

# So here I go it's my shot.
docker run \
--name worker \
-e CLOUD=gcp \
-e CORES=$CORES \
-e NAME=$NAME \
-e NAMESPACE=$NAMESPACE \
-e ACTIVATION_TOKEN=$ACTIVATION_TOKEN \
-e IP_ADDRESS=$IP_ADDRESS \
-e BATCH_LOGS_STORAGE_URI=$BATCH_LOGS_STORAGE_URI \
-e INSTANCE_ID=$INSTANCE_ID \
-e PROJECT=$PROJECT \
-e ZONE=$ZONE \
-e REGION=$REGION \
-e DOCKER_PREFIX=$DOCKER_PREFIX \
-e DOCKER_ROOT_IMAGE=$DOCKER_ROOT_IMAGE \
-e INSTANCE_CONFIG=$INSTANCE_CONFIG \
-e MAX_IDLE_TIME_MSECS=$MAX_IDLE_TIME_MSECS \
-e BATCH_WORKER_IMAGE=$BATCH_WORKER_IMAGE \
-e BATCH_WORKER_IMAGE_ID=$BATCH_WORKER_IMAGE_ID \
-e INTERNET_INTERFACE=$INTERNET_INTERFACE \
-e UNRESERVED_WORKER_DATA_DISK_SIZE_GB=$UNRESERVED_WORKER_DATA_DISK_SIZE_GB \
-e ACCEPTABLE_QUERY_JAR_URL_PREFIX=$ACCEPTABLE_QUERY_JAR_URL_PREFIX \
-e INTERNAL_GATEWAY_IP=$INTERNAL_GATEWAY_IP \
-v /var/run/docker.sock:/var/run/docker.sock \
-v /var/run/netns:/var/run/netns:shared \
-v /usr/bin/docker:/usr/bin/docker \
-v /batch:/batch:shared \
-v /logs:/logs \
-v /global-config:/global-config \
-v /deploy-config:/deploy-config \
-v /cloudfuse:/cloudfuse:shared \
-v /etc/netns:/etc/netns \
-v /sys/fs/cgroup:/sys/fs/cgroup \
--mount type=bind,source=/mnt/disks/$WORKER_DATA_DISK_NAME,target=/host \
--mount type=bind,source=/dev,target=/dev,bind-propagation=rshared \
-p 5000:5000 \
--device /dev/fuse \
--device $XFS_DEVICE \
--device /dev \
--privileged \
--cap-add SYS_ADMIN \
--security-opt apparmor:unconfined \
--network host \
$BATCH_WORKER_IMAGE \
python3 -u -m batch.worker.worker >worker.log 2>&1

[ $? -eq 0 ] || tail -n 1000 worker.log

while true; do
gcloud -q compute instances delete $NAME --zone=$ZONE
sleep 1
done
''',
                },
                {
                    'key': 'shutdown-script',
                    'value': '''
set -x

INSTANCE_ID=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/instance_id")
NAME=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google')

journalctl -u docker.service > dockerd.log
''',
                },
                {'key': 'activation_token', 'value': activation_token},
                {'key': 'batch_worker_image', 'value': BATCH_WORKER_IMAGE},
                {'key': 'docker_root_image', 'value': DOCKER_ROOT_IMAGE},
                {'key': 'docker_prefix', 'value': DOCKER_PREFIX},
                {'key': 'namespace', 'value': DEFAULT_NAMESPACE},
                {'key': 'internal_ip', 'value': INTERNAL_GATEWAY_IP},
                {'key': 'batch_logs_storage_uri', 'value': file_store.batch_logs_storage_uri},
                {'key': 'instance_id', 'value': file_store.instance_id},
                {'key': 'max_idle_time_msecs', 'value': max_idle_time_msecs},
                {'key': 'region', 'value': region},
                {
                    'key': 'instance_config',
                    'value': base64.b64encode(json.dumps(instance_config.to_dict()).encode()).decode(),
                },
            ]
        },
        'tags': {'items': ["batch2-agent"]},
    }
