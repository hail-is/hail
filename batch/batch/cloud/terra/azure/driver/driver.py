import asyncio
import base64
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Tuple

import aiohttp

from gear import Database
from gear.cloud_config import get_azure_config
from hailtop import aiotools
from hailtop.aiocloud.aioazure import AzurePricingClient
from hailtop.aiocloud.aioterra.azure import TerraClient
from hailtop.config import get_deploy_config
from hailtop.config.deploy_config import TerraDeployConfig
from hailtop.utils import periodically_call, secret_alnum_string

from .....batch_configuration import DOCKER_PREFIX, INTERNAL_GATEWAY_IP
from .....driver.driver import CloudDriver
from .....driver.instance import Instance
from .....driver.instance_collection import InstanceCollectionManager, JobPrivateInstanceManager, Pool
from .....driver.location import CloudLocationMonitor
from .....driver.resource_manager import CloudResourceManager, VMDoesNotExist, VMState, VMStateCreating
from .....file_store import FileStore
from .....inst_coll_config import InstanceCollectionConfigs
from .....instance_config import InstanceConfig, QuantifiedResource
from ....azure.driver.billing_manager import AzureBillingManager
from ....azure.resource_utils import (
    azure_machine_type_to_worker_type_and_cores,
    azure_worker_memory_per_core_mib,
    azure_worker_properties_to_machine_type,
)
from ....utils import ACCEPTABLE_QUERY_JAR_URL_PREFIX
from ..instance_config import TerraAzureSlimInstanceConfig

log = logging.getLogger('driver')

TERRA_AZURE_INSTANCE_CONFIG_VERSION = 1

deploy_config = get_deploy_config()


class SingleRegionMonitor(CloudLocationMonitor):
    @staticmethod
    async def create(default_region: str) -> 'SingleRegionMonitor':
        return SingleRegionMonitor(default_region)

    def __init__(self, default_region: str):
        self._default_region = default_region

    def default_location(self) -> str:
        return self._default_region

    def choose_location(
        self,
        cores: int,
        local_ssd_data_disk: bool,
        data_disk_size_gb: int,
        preemptible: bool,
        regions: List[str],
        machine_type: str,
    ) -> str:
        return self._default_region


def create_disk_config(
    disk_name: str,
    disk_resource_id: str,
    disk_size_gb: int,
) -> Dict[str, Any]:
    return {
        'common': {
            'name': disk_name,
            'description': disk_name,
            'cloningInstructions': 'COPY_NOTHING',
            'accessScope': 'PRIVATE_ACCESS',
            'managedBy': 'USER',
            'resourceId': disk_resource_id,
            'properties': [],
        },
        'azureDisk': {
            'name': disk_name,
            'size': disk_size_gb,
        },
    }


def create_vm_config(
    file_store: FileStore,
    location: str,
    machine_name: str,
    machine_type: str,
    activation_token: str,
    max_idle_time_msecs: int,
    instance_config: InstanceConfig,
):
    BATCH_WORKER_IMAGE = os.environ['HAIL_BATCH_WORKER_IMAGE']
    TERRA_STORAGE_ACCOUNT = os.environ['TERRA_STORAGE_ACCOUNT']
    WORKSPACE_STORAGE_CONTAINER_ID = os.environ['WORKSPACE_STORAGE_CONTAINER_ID']
    WORKSPACE_STORAGE_CONTAINER_URL = os.environ['WORKSPACE_STORAGE_CONTAINER_URL']
    WORKSPACE_MANAGER_URL = os.environ['WORKSPACE_MANAGER_URL']
    WORKSPACE_ID = os.environ['WORKSPACE_ID']

    instance_config_base64 = base64.b64encode(json.dumps(instance_config.to_dict()).encode()).decode()

    assert isinstance(deploy_config, TerraDeployConfig)
    assert isinstance(instance_config, TerraAzureSlimInstanceConfig)

    startup_script = rf"""#cloud-config

mounts:
  - [ ephemeral0, null ]
  - [ ephemeral0.1, null ]

write_files:
  - owner: batch-worker:batch-worker
    path: /startup.sh
    content: |
      #!/bin/bash

      set -ex

      function cleanup() {{
          set +x
          sleep 1000
          token=$(az account get-access-token --query accessToken --output tsv)

          VM_RESOURCE_ID={ instance_config._resource_id }
          curl -X POST "{ WORKSPACE_MANAGER_URL }/api/workspaces/v1/$WORKSPACE_ID/resources/controlled/azure/vm/$VM_RESOURCE_ID" \
              -H  "accept: */*" \
              -H  "Authorization: Bearer $token" \
              -H  "Content-Type: application/json" \
              -d "{{\"jobControl\":{{\"id\":\"$VM_RESOURCE_ID\"}}}}"
      }}

      trap cleanup EXIT

      # Install things
      apt-get update
      apt-get -o DPkg::Lock::Timeout=60 install -y \
          apt-transport-https \
          ca-certificates \
          curl \
          gnupg \
          jq \
          lsb-release \
          software-properties-common

      curl --connect-timeout 5 \
           --max-time 10 \
           --retry 5 \
           --retry-max-time 40 \
           --location \
           --fail \
           --silent \
           --show-error \
           https://download.docker.com/linux/ubuntu/gpg | apt-key add -

      add-apt-repository \
         "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
         $(lsb_release -cs) \
         stable"

      apt-get install -y docker-ce

      curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

      az login --identity --allow-no-subscription

      # avoid "unable to get current user home directory: os/user lookup failed"
      export HOME=/root

      # A safe hunch based on what was available on the ubuntu vm
      UNRESERVED_WORKER_DATA_DISK_SIZE_GB=50
      ACCEPTABLE_QUERY_JAR_URL_PREFIX={ ACCEPTABLE_QUERY_JAR_URL_PREFIX }

      sudo mkdir -p /host/batch
      sudo mkdir -p /host/logs
      sudo mkdir -p /host/cloudfuse

      sudo mkdir -p /etc/netns

      sudo mkdir /deploy-config
      sudo cat >/deploy-config/deploy-config.json <<EOF
      { json.dumps(get_deploy_config().with_location('external').get_config()) }
      EOF


      SUBSCRIPTION_ID=$(curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/compute/subscriptionId?api-version=2021-02-01&format=text")
      RESOURCE_GROUP=$(curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/compute/resourceGroupName?api-version=2021-02-01&format=text")
      LOCATION=$(curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/compute/location?api-version=2021-02-01&format=text")
      NAME=$(curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/compute/name?api-version=2021-02-01&format=text")
      IP_ADDRESS=$(curl -s -H Metadata:true --noproxy "*" "http://169.254.169.254/metadata/instance/network/interface/0/ipv4/ipAddress/0/privateIpAddress?api-version=2021-02-01&format=text")

      CORES=$(nproc)
      NAMESPACE=default
      ACTIVATION_TOKEN={ activation_token }
      BATCH_LOGS_STORAGE_URI={ file_store.batch_logs_storage_uri }
      INSTANCE_ID={ file_store.instance_id }
      INSTANCE_CONFIG="{ instance_config_base64 }"
      MAX_IDLE_TIME_MSECS={ max_idle_time_msecs }
      BATCH_WORKER_IMAGE={ BATCH_WORKER_IMAGE }
      INTERNET_INTERFACE=eth0
      WORKSPACE_STORAGE_CONTAINER_ID={ WORKSPACE_STORAGE_CONTAINER_ID }
      TERRA_STORAGE_ACCOUNT={ TERRA_STORAGE_ACCOUNT }
      WORKSPACE_STORAGE_CONTAINER_URL={ WORKSPACE_STORAGE_CONTAINER_URL }
      WORKSPACE_MANAGER_URL={ WORKSPACE_MANAGER_URL }
      WORKSPACE_ID={ WORKSPACE_ID }
      REGION={ instance_config.region_for(location) }
      INTERNAL_GATEWAY_IP={ INTERNAL_GATEWAY_IP }
      DOCKER_PREFIX={ DOCKER_PREFIX }

      # private job network = 172.20.0.0/16
      # public job network = 172.21.0.0/16
      # [all networks] Rewrite traffic coming from containers to masquerade as the host
      iptables --table nat --append POSTROUTING --source 172.20.0.0/15 --jump MASQUERADE

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
      -e BATCH_LOGS_STORAGE_URI=$BATCH_LOGS_STORAGE_URI \
      -e INSTANCE_ID=$INSTANCE_ID \
      -e SUBSCRIPTION_ID=$SUBSCRIPTION_ID \
      -e RESOURCE_GROUP=$RESOURCE_GROUP \
      -e LOCATION=$LOCATION \
      -e INSTANCE_CONFIG=$INSTANCE_CONFIG \
      -e MAX_IDLE_TIME_MSECS=$MAX_IDLE_TIME_MSECS \
      -e BATCH_WORKER_IMAGE=$BATCH_WORKER_IMAGE \
      -e BATCH_WORKER_IMAGE_ID=$BATCH_WORKER_IMAGE_ID \
      -e INTERNET_INTERFACE=$INTERNET_INTERFACE \
      -e INTERNAL_GATEWAY_IP=$INTERNAL_GATEWAY_IP \
      -e DOCKER_PREFIX=$DOCKER_PREFIX \
      -e HAIL_TERRA=true \
      -e WORKSPACE_STORAGE_CONTAINER_ID=$WORKSPACE_STORAGE_CONTAINER_ID \
      -e WORKSPACE_STORAGE_CONTAINER_URL=$WORKSPACE_STORAGE_CONTAINER_URL \
      -e TERRA_STORAGE_ACCOUNT=$TERRA_STORAGE_ACCOUNT \
      -e WORKSPACE_MANAGER_URL=$WORKSPACE_MANAGER_URL \
      -e WORKSPACE_ID=$WORKSPACE_ID \
      -e UNRESERVED_WORKER_DATA_DISK_SIZE_GB=$UNRESERVED_WORKER_DATA_DISK_SIZE_GB \
      -e ACCEPTABLE_QUERY_JAR_URL_PREFIX=$ACCEPTABLE_QUERY_JAR_URL_PREFIX \
      -e REGION=$REGION \
      -v /var/run/docker.sock:/var/run/docker.sock \
      -v /var/run/netns:/var/run/netns:shared \
      -v /usr/bin/docker:/usr/bin/docker \
      -v /usr/sbin/xfs_quota:/usr/sbin/xfs_quota \
      -v /batch:/batch:shared \
      -v /logs:/logs \
      -v /global-config:/global-config \
      -v /cloudfuse:/cloudfuse:shared \
      -v /etc/netns:/etc/netns \
      -v /sys/fs/cgroup:/sys/fs/cgroup \
      --mount type=bind,source=/host,target=/host \
      --mount type=bind,source=/dev,target=/dev,bind-propagation=rshared \
      -p 5000:5000 \
      --device /dev/fuse \
      --device /dev \
      --privileged \
      --cap-add SYS_ADMIN \
      --security-opt apparmor:unconfined \
      --network host \
      $BATCH_WORKER_IMAGE \
      python3 -u -m batch.worker.worker


runcmd:
  - nohup bash /startup.sh 2>&1 >worker.log &
    """

    encoded_startup_script = base64.b64encode(startup_script.encode()).decode()

    config = {
        'common': {
            'name': machine_name,
            'description': machine_name,
            'cloningInstructions': 'COPY_NOTHING',
            'accessScope': 'PRIVATE_ACCESS',
            'managedBy': 'USER',
            'resourceId': instance_config._resource_id,
            'properties': [],
        },
        'azureVm': {
            'name': machine_name,
            'vmSize': machine_type,
            'vmImage': {
                'publisher': 'Canonical',
                'offer': '0001-com-ubuntu-server-focal',
                'sku': '20_04-lts-gen2',
                'version': '20.04.202305150',
            },
            'vmUser': {
                'name': 'hail-admin',
                'password': secret_alnum_string(),
            },
            'ephemeralOSDisk': 'NONE',
            'customData': encoded_startup_script,
        },
        'jobControl': {
            'id': machine_name[32:],
        },
    }

    if not instance_config.local_ssd_data_disk:
        config['azureVm']['diskId'] = instance_config._disk_resource_id

    return config


class TerraAzureResourceManager(CloudResourceManager):
    def __init__(
        self,
        billing_manager,
    ):
        self.terra_client = TerraClient()
        self.billing_manager = billing_manager

    async def delete_vm(self, instance: Instance):
        config = instance.instance_config
        assert isinstance(config, TerraAzureSlimInstanceConfig)
        terra_vm_resource_id = config._resource_id

        if not config.local_ssd_data_disk:
            terra_disk_resource_id = config._disk_resource_id
            await self.terra_client.post(
                f'/disks/{terra_disk_resource_id}',
                json={
                    'jobControl': {'id': str(uuid.uuid4())},
                },
            )

        try:
            await self.terra_client.post(
                f'/vm/{terra_vm_resource_id}',
                json={
                    'jobControl': {'id': str(uuid.uuid4())},
                },
            )
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise VMDoesNotExist() from e
            raise

    async def get_vm_state(self, instance: Instance) -> VMState:
        # TODO This should look at the response and use all applicable lifecycle types
        try:
            spec = await self.terra_client.get(f'/vm/create-result/{instance.name[32:]}')
            return VMStateCreating(spec, instance.time_created)
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise VMDoesNotExist() from e
            raise

    def machine_type(self, cores: int, worker_type: str, local_ssd: bool) -> str:
        return azure_worker_properties_to_machine_type(worker_type, cores, local_ssd)

    def worker_type_and_cores(self, machine_type: str) -> Tuple[str, int]:
        return azure_machine_type_to_worker_type_and_cores(machine_type)

    def instance_config(
        self,
        machine_type: str,
        preemptible: bool,
        local_ssd_data_disk: bool,
        data_disk_size_gb: int,
        boot_disk_size_gb: int,
        job_private: bool,
        location: str,
    ) -> TerraAzureSlimInstanceConfig:
        return TerraAzureSlimInstanceConfig.create(
            self.billing_manager.product_versions,
            machine_type,
            preemptible,
            local_ssd_data_disk,
            data_disk_size_gb,
            boot_disk_size_gb,
            job_private,
            location,
        )

    def instance_config_from_dict(self, data: dict) -> TerraAzureSlimInstanceConfig:
        return TerraAzureSlimInstanceConfig.from_dict(data)

    async def create_vm(
        self,
        file_store: FileStore,
        machine_name: str,
        activation_token: str,
        max_idle_time_msecs: int,
        local_ssd_data_disk: bool,
        data_disk_size_gb: int,
        boot_disk_size_gb: int,
        preemptible: bool,
        job_private: bool,
        location: str,
        machine_type: str,
        instance_config: InstanceConfig,
    ) -> List[QuantifiedResource]:
        assert isinstance(instance_config, TerraAzureSlimInstanceConfig)
        worker_type, cores = self.worker_type_and_cores(machine_type)

        memory_mib = azure_worker_memory_per_core_mib(worker_type) * cores
        memory_in_bytes = memory_mib << 20
        cores_mcpu = cores * 1000
        total_resources_on_instance = instance_config.quantified_resources(
            cpu_in_mcpu=cores_mcpu, memory_in_bytes=memory_in_bytes, extra_storage_in_gib=0
        )

        if not local_ssd_data_disk:
            disk_name = f'{machine_name}-data'
            disk_config = create_disk_config(disk_name, instance_config._disk_resource_id, data_disk_size_gb)
            try:
                res = await self.terra_client.post('/disks', json=disk_config)
                log.info(f'Terra response creating disk {disk_name}: {res}')
            except Exception:
                log.exception(f'error while creating disk {disk_name}')
                return total_resources_on_instance

        assert location == 'eastus'
        vm_config = create_vm_config(
            file_store,
            location,
            machine_name,
            machine_type,
            activation_token,
            max_idle_time_msecs,
            instance_config,
        )

        try:
            res = await self.terra_client.post('/vm', json=vm_config)
            log.info(f'Terra response creating machine {machine_name}: {res}')
        except Exception:
            log.exception(f'error while creating machine {machine_name}')
        return total_resources_on_instance


class TerraAzureDriver(CloudDriver):
    @staticmethod
    async def create(
        app,
        db: Database,  # BORROWED
        machine_name_prefix: str,
        namespace: str,
        inst_coll_configs: InstanceCollectionConfigs,
    ) -> 'TerraAzureDriver':
        azure_config = get_azure_config()
        region = azure_config.region
        regions = [region]

        region_args = [(r,) for r in regions]
        await db.execute_many(
            """
INSERT INTO regions (region) VALUES (%s)
ON DUPLICATE KEY UPDATE region = region;
""",
            region_args,
        )

        db_regions = {
            record['region']: record['region_id']
            async for record in db.select_and_fetchall('SELECT region_id, region from regions')
        }
        assert max(db_regions.values()) < 64, str(db_regions)
        app['regions'] = db_regions

        region_monitor = await SingleRegionMonitor.create(region)
        inst_coll_manager = InstanceCollectionManager(db, machine_name_prefix, region_monitor, region, regions)
        pricing_client = AzurePricingClient()
        billing_manager = await AzureBillingManager.create(db, pricing_client, regions)
        resource_manager = TerraAzureResourceManager(billing_manager)
        task_manager = aiotools.BackgroundTaskManager()
        task_manager.ensure_future(periodically_call(300, billing_manager.refresh_resources_from_retail_prices))

        create_pools_coros = [
            Pool.create(
                app,
                db,
                inst_coll_manager,
                resource_manager,
                machine_name_prefix,
                config,
                app['async_worker_pool'],
                task_manager,
            )
            for config in inst_coll_configs.name_pool_config.values()
        ]

        jpim, *_ = await asyncio.gather(
            JobPrivateInstanceManager.create(
                app,
                db,
                inst_coll_manager,
                resource_manager,
                machine_name_prefix,
                inst_coll_configs.jpim_config,
                task_manager,
            ),
            *create_pools_coros,
        )

        return TerraAzureDriver(
            db,
            machine_name_prefix,
            namespace,
            region_monitor,
            inst_coll_manager,
            jpim,
            billing_manager,
            task_manager,
        )

    def __init__(
        self,
        db: Database,
        machine_name_prefix: str,
        namespace: str,
        region_monitor: SingleRegionMonitor,
        inst_coll_manager: InstanceCollectionManager,
        job_private_inst_manager: JobPrivateInstanceManager,
        billing_manager: AzureBillingManager,
        task_manager: aiotools.BackgroundTaskManager,
    ):
        self.db = db
        self.machine_name_prefix = machine_name_prefix
        self.namespace = namespace
        self.region_monitor = region_monitor
        self.job_private_inst_manager = job_private_inst_manager
        self._inst_coll_manager = inst_coll_manager
        self._billing_manager = billing_manager
        self._task_manager = task_manager

    @property
    def billing_manager(self) -> AzureBillingManager:
        return self._billing_manager

    @property
    def inst_coll_manager(self) -> InstanceCollectionManager:
        return self._inst_coll_manager

    async def shutdown(self) -> None:
        await self._task_manager.shutdown_and_wait()

    def get_quotas(self):
        raise NotImplementedError
