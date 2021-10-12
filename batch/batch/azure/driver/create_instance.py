import os
import logging
import base64
import json

from ...batch_configuration import DOCKER_ROOT_IMAGE, DOCKER_PREFIX, DEFAULT_NAMESPACE, AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_SSH_PUBLIC_KEY
from ...file_store import FileStore
from ...resource_utils import unreserved_worker_data_disk_size_gib

from ..resource_utils import azure_machine_type_to_worker_type_cores_local_ssd, local_ssd_size
from ..instance_config import AzureInstanceConfig

log = logging.getLogger('create_instance')

BATCH_WORKER_IMAGE = os.environ['HAIL_BATCH_WORKER_IMAGE']
INTERNAL_GATEWAY_IP = os.environ['HAIL_INTERNAL_IP']

log.info(f'BATCH_WORKER_IMAGE {BATCH_WORKER_IMAGE}')


def create_instance_config(
    app,
    location,
    machine_name,
    machine_type,
    activation_token,
    max_idle_time_msecs,
    local_ssd_data_disk,
    external_data_disk_size_gb,
    preemptible,
    job_private,
) -> AzureInstanceConfig:
    file_store: FileStore = app['file_store']

    worker_type, cores, _ = azure_machine_type_to_worker_type_cores_local_ssd(machine_type)

    if job_private:
        if local_ssd_data_disk:
            unreserved_disk_storage_gb = local_ssd_size(worker_type, cores)
        else:
            unreserved_disk_storage_gb = external_data_disk_size_gb
    else:
        unreserved_disk_storage_gb = unreserved_worker_data_disk_size_gib(
            'azure', local_ssd_data_disk, external_data_disk_size_gb, cores, worker_type,
        )
    assert unreserved_disk_storage_gb >= 0

    worker_data_disk_name = 'sdb'

    if local_ssd_data_disk:
        data_disks = []
    else:
        data_disks = [
            {
                "name": "[concat(parameters('vmName'), '-data')]",
                "lun": 1,  # because this is 1, the data disk will always be at 'sdb'
                "managedDisk": {
                    "storageAccountType": "Standard_LRS"
                },
                "createOption": "Empty",
                "diskSizeGB": external_data_disk_size_gb,
                "deleteOption": None
            }
        ]

    startup_script = r'''#!/bin/sh
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
'''
    startup_script = base64.b64encode(startup_script.encode('utf-8')).decode('utf-8')

    run_script = f'''
#!/bin/bash
set -x

WORKER_DATA_DISK_NAME="{worker_data_disk_name}"
UNRESERVED_WORKER_DATA_DISK_SIZE_GB="{unreserved_disk_storage_gb}"

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

sudo mkdir -p /etc/netns

curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/compute/userData?api-version=2021-02-01&format=text" | \
  base64 --decode > userdata

SUBSCRIPTION_ID=$(curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/compute/subscriptionId?api-version=2021-02-01&format=text")
RESOURCE_GROUP=$(curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/compute/resourceGroupName?api-version=2021-02-01&format=text")

CORES=$(nproc)
NAMESPACE=$(jq -r '.namespace' userdata)
ACTIVATION_TOKEN=$(jq -r '.activation_token' userdata)
IP_ADDRESS=$(curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/network/interface/0/ipv4/ipAddress/0/publicIpAddress?api-version=2021-02-01&format=text")

BATCH_LOGS_BUCKET_NAME=$(jq -r '.batch_logs_bucket_name' userdata)
INSTANCE_ID=$(jq -r '.instance_id' userdata)
INSTANCE_CONFIG=$(jq -r '.instance_config' userdata)
MAX_IDLE_TIME_MSECS=$(jq -r '.max_idle_time_msecs' userdata)
NAME=$(curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/compute/name?api-version=2021-02-01&format=text")

BATCH_WORKER_IMAGE=$(jq -r '.batch_worker_image' userdata)
DOCKER_ROOT_IMAGE=$(jq -r '.docker_root_image' userdata)
DOCKER_PREFIX=$(jq -r '.docker_prefix' userdata)

INTERNAL_GATEWAY_IP=$(jq -r '.internal_gateway_ip' userdata)

# private job network = 10.0.0.0/16
# public job network = 10.1.0.0/16
# [all networks] Rewrite traffic coming from containers to masquerade as the host
sudo iptables --table nat --append POSTROUTING --source 10.0.0.0/15 --jump MASQUERADE

# [public]
# Block public traffic to the metadata server
sudo iptables --append FORWARD --source 10.1.0.0/16 --destination 169.254.169.254 --jump DROP
# But allow the internal gateway
sudo iptables --append FORWARD --destination $INTERNAL_GATEWAY_IP --jump ACCEPT
# And this worker
sudo iptables --append FORWARD --destination $IP_ADDRESS --jump ACCEPT
# Forbid outgoing requests to cluster-internal IP addresses
ENS_DEVICE=$(ip link list | grep ens | awk -F": " '{{ print $2 }}')
sudo iptables --append FORWARD --out-interface $ENS_DEVICE ! --destination 10.128.0.0/16 --jump ACCEPT

# retry once
docker pull $BATCH_WORKER_IMAGE || \
(echo 'pull failed, retrying' && sleep 15 && docker pull $BATCH_WORKER_IMAGE)

BATCH_WORKER_IMAGE_ID=$(docker inspect $BATCH_WORKER_IMAGE --format='{{{{.Id}}}}' | cut -d':' -f2)

# So here I go it's my shot.
docker run \
-e CLOUD=azure \
-e CORES=$CORES \
-e NAME=$NAME \
-e NAMESPACE=$NAMESPACE \
-e ACTIVATION_TOKEN=$ACTIVATION_TOKEN \
-e IP_ADDRESS=$IP_ADDRESS \
-e BATCH_LOGS_BUCKET_NAME=$BATCH_LOGS_BUCKET_NAME \
-e INSTANCE_ID=$INSTANCE_ID \
-e SUBSCRIPTION_ID=$SUBSCRIPTION_ID \
-e RESOURCE_GROUP=$RESOURCE_GROUP \
-e DOCKER_PREFIX=$DOCKER_PREFIX \
-e DOCKER_ROOT_IMAGE=$DOCKER_ROOT_IMAGE \
-e INSTANCE_CONFIG=$INSTANCE_CONFIG \
-e MAX_IDLE_TIME_MSECS=$MAX_IDLE_TIME_MSECS \
-e BATCH_WORKER_IMAGE=$BATCH_WORKER_IMAGE \
-e BATCH_WORKER_IMAGE_ID=$BATCH_WORKER_IMAGE_ID \
-e UNRESERVED_WORKER_DATA_DISK_SIZE_GB=$UNRESERVED_WORKER_DATA_DISK_SIZE_GB \
-e INTERNAL_GATEWAY_IP=$INTERNAL_GATEWAY_IP \
-v /var/run/docker.sock:/var/run/docker.sock \
-v /var/run/netns:/var/run/netns:shared \
-v /usr/bin/docker:/usr/bin/docker \
-v /usr/sbin/xfs_quota:/usr/sbin/xfs_quota \
-v /batch:/batch:shared \
-v /logs:/logs \
-v /gcsfuse:/gcsfuse:shared \
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
        'batch_logs_bucket_name': file_store.batch_logs_bucket_name,
        'instance_id': file_store.instance_id,
        'max_idle_time_msecs': max_idle_time_msecs,
    }
    user_data = base64.b64encode(json.dumps(user_data).encode('utf-8')).decode('utf-8')

    tags = {
        'namespace': 'default',
        'batch-worker': '1'
    }

    vm_config = {
        "apiVersion": "2021-03-01",
        "type": "ds",
        "name": "[parameters('vmName')]",
        "location": "[parameters('location')]",
        "identity": {
            "type": "UserAssigned",
            "userAssignedIdentities": {
                "[resourceId('Microsoft.ManagedIdentity/userAssignedIdentities', parameters('userAssignedIdentityName'))]": {}
            }
        },
        "tags": {

        },
        "dependsOn": [
            "[concat('Microsoft.Network/networkInterfaces/', variables('nicName'))]"
        ],
        "properties": {
            "hardwareProfile": {
                "vmSize": machine_type
            },
            "networkProfile": {
                "networkInterfaces": [
                    {
                        "id": "[resourceId('Microsoft.Network/networkInterfaces', variables('nicName'))]",
                        "properties": {
                            "deleteOption": None
                        }
                    }
                ]
            },
            "storageProfile": {
                "osDisk": {
                    "name": "[concat(parameters('vmName'), '-os')]",
                    "createOption": "FromImage",
                    "deleteOption": 'Delete',
                    "caching": "ReadWrite",
                    "managedDisk": {
                        "storageAccountType": "Standard_LRS"
                    }
                },
                "imageReference": "[parameters('imageReference')]",
                "dataDisks": data_disks
            },
            "osProfile": {
                "computerName": "[parameters('vmName')]",
                "adminUsername": "[parameters('adminUsername')]",
                "customData": "[parameters('startupScript')]",
                "linuxConfiguration": {
                    "disablePasswordAuthentication": True,
                    "ssh": {
                        "publicKeys": [
                            {
                                "keyData": "[parameters('sshKey')]",
                                "path": "[concat('/home/', parameters('adminUsername'), '/.ssh/authorized_keys')]"
                            }
                        ]
                    }
                }
            },
            "userData": "[parameters('userData')]"
        }
    }

    if preemptible:
        vm_config['properties']['priority'] = 'Spot'
        vm_config['properties']['evictionPolicy'] = 'Delete'
        vm_config['properties']['billingProfile'] = {'maxPrice': -1}
    else:
        vm_config['properties']['priority'] = 'Regular'

    config = {
        'properties': {
            'mode': 'Incremental',
            'parameters': {
                'location': {
                    'value': location
                },
                'vmName': {
                    'value': machine_name
                },
                'sshKey': {
                    'value': AZURE_SSH_PUBLIC_KEY
                },
                'subnetId': {
                    'value': f'/subscriptions/{AZURE_SUBSCRIPTION_ID}/resourceGroups/{AZURE_RESOURCE_GROUP}/providers/Microsoft.Network/virtualNetworks/{AZURE_RESOURCE_GROUP}-vnet/subnets/default'  # FIXME: is this correct with terraform?
                },
                'adminUsername': {
                    'value': 'admin'
                },
                'userAssignedIdentityName': {
                    'value': 'batch-worker'
                },
                'startupScript': {
                    'value': startup_script
                },
                'userData': {
                    'value': user_data
                },
                'imageReference': {
                    'value': {
                        'id': f'/subscriptions/{AZURE_SUBSCRIPTION_ID}/resourceGroups/{AZURE_RESOURCE_GROUP}/providers/'
                              f'Microsoft.Compute/galleries/batch/images/batch-worker-generalized/versions/1.0.12'
                    }
                }
            },
            'template': {
                "$schema": "https://schema.management.azure.com/schemas/2015-01-01/deploymentTemplate.json#",
                "contentVersion": "1.0.0.0",
                "parameters": {
                    "location": {
                        "type": "string",
                        "defaultValue": "[resourceGroup().location]"
                    },
                    "vmName": {
                        "type": "string"
                    },
                    "sshKey": {
                        "type": "securestring"
                    },
                    "subnetId": {
                        "type": "string"
                    },
                    "adminUsername": {
                        "type": "string",
                        "defaultValue": "admin"
                    },
                    "userAssignedIdentityName": {
                        "type": "string",
                        "defaultValue": "batch-worker"
                    },
                    "startupScript": {
                        "type": "string"
                    },
                    "userData": {
                        "type": "string"
                    },
                    "imageReference": {
                        "type": "object",
                        "defaultValue":
                            {
                                "publisher": "Canonical",
                                "offer": "UbuntuServer",
                                "sku": "18.04-LTS",
                                "version": "latest"
                            }
                    }
                },
                "variables": {
                    "nsgName": "[concat(parameters('vmName'), '-nsg')]",
                    "ipName": "[concat(parameters('vmName'), '-ip')]",
                    "nicName": "[concat(parameters('vmName'), '-nic')]",
                    "ipconfigName": "[concat(parameters('vmName'), '-ipconfig')]"
                },
                "resources": [
                    {
                        "type": "Microsoft.Network/networkSecurityGroups",
                        "name": "[variables('nsgName')]",
                        "apiVersion": "2015-06-15",
                        "location": "[parameters('location')]",
                        "tags": tags,
                        "dependsOn": [],
                        "properties": {
                            "securityRules": [
                                {
                                    "name": "default-allow-ssh",
                                    "properties": {
                                        "protocol": "Tcp",
                                        "sourcePortRange": "*",
                                        "destinationPortRange": "22",
                                        "sourceAddressPrefix": "*",
                                        "destinationAddressPrefix": "*",
                                        "access": "Allow",
                                        "priority": 1000,
                                        "direction": "Inbound"
                                    }
                                }
                            ]
                        }
                    },
                    {
                        "apiVersion": "2018-01-01",
                        "type": "Microsoft.Network/publicIPAddresses",
                        "name": "[variables('ipName')]",
                        "location": "[parameters('location')]",
                        "tags": tags,
                        "dependsOn": [],
                        "properties": {
                            "publicIPAllocationMethod": "Static"
                        }
                    },
                    {
                        "apiVersion": "2015-06-15",
                        "type": "Microsoft.Network/networkInterfaces",
                        "name": "[variables('nicName')]",
                        "location": "[parameters('location')]",
                        "tags": tags,
                        "dependsOn": [
                            "[concat('Microsoft.Network/networkSecurityGroups/', variables('nsgName'))]",
                            "[concat('Microsoft.Network/publicIPAddresses/', variables('ipName'))]"
                        ],
                        "properties": {
                            "ipConfigurations": [
                                {
                                    "name": "[variables('ipconfigName')]",
                                    "properties": {
                                        "publicIPAddress": {
                                            "id": "[resourceId('Microsoft.Network/publicIpAddresses', variables('ipName'))]"
                                        },
                                        "privateIPAllocationMethod": "Dynamic",
                                        "subnet": {
                                            "id": "[parameters('subnetId')]"
                                        }
                                    }
                                }
                            ],
                            "networkSecurityGroup": {
                                "id": "[resourceId('Microsoft.Network/networkSecurityGroups', variables('nsgName'))]"
                            }
                        }
                    },
                    vm_config
                ],
                "outputs": {}
            }
        }
    }

    return AzureInstanceConfig.from_vm_config(config, job_private)
