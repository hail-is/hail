import base64
import json
import logging
import os
from shlex import quote as shq
from typing import Any, Dict, List, Optional

from gear.cloud_config import get_global_config
from hailtop.config import get_deploy_config

from ....batch_configuration import DEFAULT_NAMESPACE, DOCKER_PREFIX, DOCKER_ROOT_IMAGE, INTERNAL_GATEWAY_IP
from ....file_store import FileStore
from ....instance_config import InstanceConfig
from ...resource_utils import unreserved_worker_data_disk_size_gib
from ...utils import ACCEPTABLE_QUERY_JAR_URL_PREFIX
from ..resource_utils import azure_machine_type_to_worker_type_and_cores

log = logging.getLogger('create_instance')

BATCH_WORKER_IMAGE = os.environ['HAIL_BATCH_WORKER_IMAGE']


log.info(f'BATCH_WORKER_IMAGE {BATCH_WORKER_IMAGE}')
log.info(f'ACCEPTABLE_QUERY_JAR_URL_PREFIX {ACCEPTABLE_QUERY_JAR_URL_PREFIX}')


def create_vm_config(
    file_store: FileStore,
    resource_rates: Dict[str, float],
    location: str,
    machine_name: str,
    machine_type: str,
    activation_token: str,
    max_idle_time_msecs: int,
    local_ssd_data_disk: bool,
    data_disk_size_gb: int,
    preemptible: bool,
    job_private: bool,
    subscription_id: str,
    resource_group: str,
    ssh_public_key: str,
    max_price: Optional[float],
    instance_config: InstanceConfig,
    feature_flags: dict,
) -> dict:
    _, cores = azure_machine_type_to_worker_type_and_cores(machine_type)

    hail_azure_oauth_scope = os.environ['HAIL_AZURE_OAUTH_SCOPE']
    region = instance_config.region_for(location)

    if max_price is not None and not preemptible:
        raise ValueError(f'max price given for a nonpreemptible machine {max_price}')

    if job_private:
        unreserved_disk_storage_gb = data_disk_size_gb
    else:
        unreserved_disk_storage_gb = unreserved_worker_data_disk_size_gib(data_disk_size_gb, cores)
    assert unreserved_disk_storage_gb >= 0

    worker_data_disk_name = 'data-disk'

    if local_ssd_data_disk:
        data_disks = []
        disk_location = '/dev/disk/azure/resource'
    else:
        data_disks = [
            {
                "name": "[concat(parameters('vmName'), '-data')]",
                "lun": 2,  # because this is 2, the data disk will always be at 'sdc'
                "managedDisk": {"storageAccountType": "Premium_LRS"},
                "createOption": "Empty",
                "diskSizeGB": data_disk_size_gb,
                "deleteOption": 'Delete',
            }
        ]
        disk_location = '/dev/disk/azure/scsi1/lun2'

    make_global_config = ['mkdir /global-config']
    global_config = get_global_config()
    for name, value in global_config.items():
        make_global_config.append(f'echo -n {shq(value)} > /global-config/{name}')
    make_global_config_str = '\n'.join(make_global_config)

    assert instance_config.is_valid_configuration(resource_rates.keys())

    touch_commands: List[str] = []
    for jvm_cores in (1, 2, 4, 8):
        for _ in range(cores // jvm_cores):
            idx = len(touch_commands)
            log_path = f'/batch/jvm-container-logs/jvm-{idx}.log'
            touch_commands.append(f'sudo touch {log_path}')

    jvm_touch_command = '\n'.join(touch_commands)

    startup_script = r'''#cloud-config

mounts:
  - [ ephemeral0, null ]
  - [ ephemeral0.1, null ]

write_files:
  - owner: batch-worker:batch-worker
    path: /startup.sh
    content: |
      #!/bin/sh
      set -ex
      RESOURCE_GROUP=$(curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/compute/resourceGroupName?api-version=2021-02-01&format=text")
      NAME=$(curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/compute/name?api-version=2021-02-01&format=text")
      if [ -f "/started" ]; then
          echo "instance $NAME has previously been started"
          while true; do
          az vm delete -g $RESOURCE_GROUP -n $NAME --yes
          sleep 1
          done
          exit
      else
          touch /started
      fi
      curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/compute/userData?api-version=2021-02-01&format=text" | \
        base64 --decode | \
        jq -r '.run_script' > ./run.sh
      nohup /bin/bash run.sh >run.log 2>&1 &

runcmd:
  - sh /startup.sh
'''
    startup_script = base64.b64encode(startup_script.encode('utf-8')).decode('utf-8')

    run_script = f'''
#!/bin/bash
set -x

WORKER_DATA_DISK_NAME="{worker_data_disk_name}"
UNRESERVED_WORKER_DATA_DISK_SIZE_GB="{unreserved_disk_storage_gb}"
ACCEPTABLE_QUERY_JAR_URL_PREFIX="{ACCEPTABLE_QUERY_JAR_URL_PREFIX}"

# format worker data disk
sudo mkfs.xfs -f -m reflink=1 -n ftype=1 {disk_location}
sudo mkdir -p /mnt/disks/$WORKER_DATA_DISK_NAME
sudo mount -o prjquota {disk_location} /mnt/disks/$WORKER_DATA_DISK_NAME
sudo chmod a+w /mnt/disks/$WORKER_DATA_DISK_NAME
XFS_DEVICE=$(xfs_info /mnt/disks/$WORKER_DATA_DISK_NAME | head -n 1 | awk '{{ print $1 }}' | awk  'BEGIN {{ FS = "=" }}; {{ print $2 }}')

# reconfigure docker to use data disk
sudo service docker stop
sudo mv /var/lib/docker /mnt/disks/$WORKER_DATA_DISK_NAME/docker
sudo ln -s /mnt/disks/$WORKER_DATA_DISK_NAME/docker /var/lib/docker
sudo service docker start

# reconfigure /batch and /logs to use data disk
sudo mkdir -p /mnt/disks/$WORKER_DATA_DISK_NAME/batch/
sudo ln -s /mnt/disks/$WORKER_DATA_DISK_NAME/batch /batch

sudo mkdir -p /batch/jvm-container-logs/
{jvm_touch_command}

sudo mkdir -p /mnt/disks/$WORKER_DATA_DISK_NAME/logs/
sudo ln -s /mnt/disks/$WORKER_DATA_DISK_NAME/logs /logs

sudo mkdir -p /mnt/disks/$WORKER_DATA_DISK_NAME/cloudfuse/
sudo ln -s /mnt/disks/$WORKER_DATA_DISK_NAME/cloudfuse /cloudfuse

# Forward syslog logs to Log Analytics Agent
cat >>/etc/rsyslog.d/95-omsagent.conf <<EOF
kern.warning       @127.0.0.1:25224
user.warning       @127.0.0.1:25224
daemon.warning     @127.0.0.1:25224
auth.warning       @127.0.0.1:25224
uucp.warning       @127.0.0.1:25224
authpriv.warning   @127.0.0.1:25224
ftp.warning        @127.0.0.1:25224
cron.warning       @127.0.0.1:25224
local0.warning     @127.0.0.1:25224
local1.warning     @127.0.0.1:25224
local2.warning     @127.0.0.1:25224
local3.warning     @127.0.0.1:25224
local4.warning     @127.0.0.1:25224
local5.warning     @127.0.0.1:25224
local6.warning     @127.0.0.1:25224
local7.warning     @127.0.0.1:25224
EOF

sudo service rsyslog restart

sudo mkdir -p /etc/netns

curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/compute/userData?api-version=2021-02-01&format=text" | \
  base64 --decode > userdata

SUBSCRIPTION_ID=$(curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/compute/subscriptionId?api-version=2021-02-01&format=text")
RESOURCE_GROUP=$(curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/compute/resourceGroupName?api-version=2021-02-01&format=text")
LOCATION=$(curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/compute/location?api-version=2021-02-01&format=text")

CORES=$(nproc)
NAMESPACE=$(jq -r '.namespace' userdata)
ACTIVATION_TOKEN=$(jq -r '.activation_token' userdata)
IP_ADDRESS=$(curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/network/interface/0/ipv4/ipAddress/0/privateIpAddress?api-version=2021-02-01&format=text")

BATCH_LOGS_STORAGE_URI=$(jq -r '.batch_logs_storage_uri' userdata)
INSTANCE_ID=$(jq -r '.instance_id' userdata)
INSTANCE_CONFIG=$(jq -r '.instance_config' userdata)
MAX_IDLE_TIME_MSECS=$(jq -r '.max_idle_time_msecs' userdata)
NAME=$(curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/compute/name?api-version=2021-02-01&format=text")

BATCH_WORKER_IMAGE=$(jq -r '.batch_worker_image' userdata)
DOCKER_ROOT_IMAGE=$(jq -r '.docker_root_image' userdata)
DOCKER_PREFIX=$(jq -r '.docker_prefix' userdata)
REGION=$(jq -r '.region' userdata)
HAIL_AZURE_OAUTH_SCOPE=$(jq -r '.hail_azure_oauth_scope' userdata)

INTERNAL_GATEWAY_IP=$(jq -r '.internal_ip' userdata)

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
# Forbid outgoing requests to cluster-internal IP addresses
INTERNET_INTERFACE=eth0
iptables --append FORWARD --out-interface $INTERNET_INTERFACE ! --destination 10.128.0.0/16 --jump ACCEPT

cat >> /etc/hosts <<EOF
$INTERNAL_GATEWAY_IP batch-driver.hail
$INTERNAL_GATEWAY_IP batch.hail
$INTERNAL_GATEWAY_IP internal.hail
EOF

{make_global_config_str}

mkdir /deploy-config
cat >/deploy-config/deploy-config.json <<EOF
{ json.dumps(get_deploy_config().with_location('gce').get_config()) }
EOF

# retry once
az acr login --name $DOCKER_PREFIX
docker pull $BATCH_WORKER_IMAGE || \
(echo 'pull failed, retrying' && sleep 15 && docker pull $BATCH_WORKER_IMAGE)

BATCH_WORKER_IMAGE_ID=$(docker inspect $BATCH_WORKER_IMAGE --format='{{{{.Id}}}}' | cut -d':' -f2)

# So here I go it's my shot.
docker run \
--name worker \
-e CLOUD=azure \
-e CORES=$CORES \
-e NAME=$NAME \
-e NAMESPACE=$NAMESPACE \
-e ACTIVATION_TOKEN=$ACTIVATION_TOKEN \
-e IP_ADDRESS=$IP_ADDRESS \
-e BATCH_LOGS_STORAGE_URI=$BATCH_LOGS_STORAGE_URI \
-e INSTANCE_ID=$INSTANCE_ID \
-e SUBSCRIPTION_ID=$SUBSCRIPTION_ID \
-e RESOURCE_GROUP=$RESOURCE_GROUP \
-e LOCATION=$LOCATION \
-e REGION=$REGION \
-e HAIL_AZURE_OAUTH_SCOPE=$HAIL_AZURE_OAUTH_SCOPE \
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
--cgroupns host \
--oom-score-adj 0 \
--oom-kill-disable \
$BATCH_WORKER_IMAGE \
python3 -u -m batch.worker.worker >worker.log 2>&1

[ $? -eq 0 ] || tail -n 1000 worker.log

while true; do
az vm delete -g $RESOURCE_GROUP -n $NAME --yes
sleep 1
done
'''

    user_data = {
        'run_script': run_script,
        'activation_token': activation_token,
        'batch_worker_image': BATCH_WORKER_IMAGE,
        'docker_root_image': DOCKER_ROOT_IMAGE,
        'docker_prefix': DOCKER_PREFIX,
        'namespace': DEFAULT_NAMESPACE,
        'internal_ip': INTERNAL_GATEWAY_IP,
        'batch_logs_storage_uri': file_store.batch_logs_storage_uri,
        'instance_id': file_store.instance_id,
        'max_idle_time_msecs': max_idle_time_msecs,
        'instance_config': base64.b64encode(json.dumps(instance_config.to_dict()).encode()).decode(),
        'region': region,
        'hail_azure_oauth_scope': hail_azure_oauth_scope,
    }
    user_data_str = base64.b64encode(json.dumps(user_data).encode('utf-8')).decode('utf-8')

    tags = {'namespace': DEFAULT_NAMESPACE, 'batch-worker': '1'}

    vm_resources = []

    if feature_flags['oms_agent']:
        vm_resources.append(
            {
                'apiVersion': '2018-06-01',
                'type': 'extensions',
                'name': 'OMSExtension',
                'location': "[parameters('location')]",
                'tags': tags,
                'dependsOn': ["[concat('Microsoft.Compute/virtualMachines/', parameters('vmName'))]"],
                'properties': {
                    'publisher': 'Microsoft.EnterpriseCloud.Monitoring',
                    'type': 'OmsAgentForLinux',
                    'typeHandlerVersion': '1.13',
                    'autoUpgradeMinorVersion': False,
                    'enableAutomaticUpgrade': False,
                    'settings': {
                        'workspaceId': "[reference(resourceId('Microsoft.OperationalInsights/workspaces/', parameters('workspaceName')), '2015-03-20').customerId]"
                    },
                    'protectedSettings': {
                        'workspaceKey': "[listKeys(resourceId('Microsoft.OperationalInsights/workspaces/', parameters('workspaceName')), '2015-03-20').primarySharedKey]"
                    },
                },
            },
        )

    vm_config: Dict[str, Any] = {
        'apiVersion': '2021-03-01',
        'type': 'Microsoft.Compute/virtualMachines',
        'name': "[parameters('vmName')]",
        'location': "[parameters('location')]",
        'identity': {
            'type': 'UserAssigned',
            'userAssignedIdentities': {
                "[resourceId('Microsoft.ManagedIdentity/userAssignedIdentities', parameters('userAssignedIdentityName'))]": {}
            },
        },
        'tags': tags,
        'dependsOn': ["[concat('Microsoft.Network/networkInterfaces/', variables('nicName'))]"],
        'properties': {
            'hardwareProfile': {'vmSize': machine_type},
            'networkProfile': {
                'networkInterfaces': [
                    {
                        'id': "[resourceId('Microsoft.Network/networkInterfaces', variables('nicName'))]",
                        'properties': {'deleteOption': 'Delete'},
                    }
                ]
            },
            'storageProfile': {
                'osDisk': {
                    'name': "[concat(parameters('vmName'), '-os')]",
                    'createOption': 'FromImage',
                    'deleteOption': 'Delete',
                    'caching': 'ReadOnly',
                    'managedDisk': {'storageAccountType': 'Standard_LRS'},
                },
                'imageReference': "[parameters('imageReference')]",
                'dataDisks': data_disks,
            },
            'osProfile': {
                'computerName': "[parameters('vmName')]",
                'adminUsername': "[parameters('adminUsername')]",
                'customData': "[parameters('startupScript')]",
                'linuxConfiguration': {
                    'disablePasswordAuthentication': True,
                    'ssh': {
                        'publicKeys': [
                            {
                                'keyData': "[parameters('sshKey')]",
                                'path': "[concat('/home/', parameters('adminUsername'), '/.ssh/authorized_keys')]",
                            }
                        ]
                    },
                },
            },
            'userData': "[parameters('userData')]",
        },
        'resources': vm_resources,
    }

    properties = vm_config['properties']
    if preemptible:
        properties['priority'] = 'Spot'
        properties['evictionPolicy'] = 'Delete'
        properties['billingProfile'] = {'maxPrice': max_price if max_price is not None else -1}
    else:
        properties['priority'] = 'Regular'

    return {
        'tags': tags,
        'properties': {
            'mode': 'Incremental',
            'parameters': {
                'location': {'value': location},
                'vmName': {'value': machine_name},
                'sshKey': {'value': ssh_public_key},
                'subnetId': {
                    'value': f'/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Network/virtualNetworks/default/subnets/batch-worker-subnet'
                },
                'adminUsername': {'value': 'batch-worker'},
                'userAssignedIdentityName': {'value': 'batch-worker'},
                'startupScript': {'value': startup_script},
                'userData': {'value': user_data_str},
                'imageReference': {
                    'value': {
                        'id': f'/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/'
                        f'Microsoft.Compute/galleries/{resource_group}_batch/images/batch-worker-22-04/versions/0.0.13'
                    }
                },
                'workspaceName': {
                    'value': f'{resource_group}-logs',
                },
            },
            'template': {
                '$schema': 'https://schema.management.azure.com/schemas/2015-01-01/deploymentTemplate.json#',
                'contentVersion': '1.0.0.0',
                'parameters': {
                    'location': {'type': 'string', 'defaultValue': '[resourceGroup().location]'},
                    'vmName': {'type': 'string'},
                    'sshKey': {'type': 'securestring'},
                    'subnetId': {'type': 'string'},
                    'adminUsername': {'type': 'string', 'defaultValue': 'admin'},
                    'userAssignedIdentityName': {'type': 'string', 'defaultValue': 'batch-worker'},
                    'startupScript': {'type': 'string'},
                    'userData': {'type': 'string'},
                    'imageReference': {
                        'type': 'object',
                        'defaultValue': {
                            'publisher': 'Canonical',
                            'offer': 'UbuntuServer',
                            'sku': '18.04-LTS',
                            'version': 'latest',
                        },
                    },
                    'workspaceName': {'type': 'string'},
                },
                'variables': {
                    'ipName': "[concat(parameters('vmName'), '-ip')]",
                    'nicName': "[concat(parameters('vmName'), '-nic')]",
                    'ipconfigName': "[concat(parameters('vmName'), '-ipconfig')]",
                },
                'resources': [
                    {
                        'apiVersion': '2018-01-01',
                        'type': 'Microsoft.Network/publicIPAddresses',
                        'name': "[variables('ipName')]",
                        'location': "[parameters('location')]",
                        'tags': tags,
                        'dependsOn': [],
                        'properties': {'publicIPAllocationMethod': 'Static'},
                    },
                    {
                        'apiVersion': '2015-06-15',
                        'type': 'Microsoft.Network/networkInterfaces',
                        'name': "[variables('nicName')]",
                        'location': "[parameters('location')]",
                        'tags': tags,
                        'dependsOn': ["[concat('Microsoft.Network/publicIPAddresses/', variables('ipName'))]"],
                        'properties': {
                            'ipConfigurations': [
                                {
                                    'name': "[variables('ipconfigName')]",
                                    'properties': {
                                        'publicIPAddress': {
                                            'id': "[resourceId('Microsoft.Network/publicIpAddresses', variables('ipName'))]",
                                            'properties': {'deleteOption': 'Delete'},
                                        },
                                        'privateIPAllocationMethod': 'Dynamic',
                                        'subnet': {'id': "[parameters('subnetId')]"},
                                    },
                                }
                            ],
                            'networkSecurityGroup': {
                                'id': f'/subscriptions/{subscription_id}/resourceGroups/{resource_group}'
                                f'/providers/Microsoft.Network/networkSecurityGroups/batch-worker-nsg'
                            },
                        },
                    },
                    vm_config,
                ],
                'outputs': {},
            },
        },
    }
