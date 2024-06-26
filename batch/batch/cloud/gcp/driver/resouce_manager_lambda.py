import logging
import uuid
from typing import List
import os
import requests

import aiohttp

from hailtop.aiocloud import aiogoogle
from hailtop.utils.time import parse_timestamp_msecs

from ....driver.instance import Instance
from ....driver.resource_manager import (
    CloudResourceManager,
    UnknownVMState,
    VMDoesNotExist,
    VMState,
    VMStateCreating,
    VMStateRunning,
    VMStateTerminated,
)
from ....file_store import FileStore
from ....instance_config import InstanceConfig, QuantifiedResource
from ..instance_config import GCPSlimInstanceConfig, LambdaSlimInstanceConfig
from ..resource_utils import (
    GCP_MACHINE_FAMILY,
    family_worker_type_cores_to_gcp_machine_type,
    gcp_machine_type_to_cores_and_memory_bytes,
)
from .billing_manager import GCPBillingManager
from .create_instance import create_vm_config

log = logging.getLogger('resource_manager')


class LambdaResourceManager(CloudResourceManager):
    def __init__(
        self,
        project: str,
        compute_client: aiogoogle.GoogleComputeClient,  # BORROWED
        billing_manager: GCPBillingManager,  # BORROWED
    ):
        self.compute_client = compute_client
        self.project = project
        self.billing_manager = billing_manager

    async def delete_vm(self, instance: Instance):
        raise NotImplementedError

    async def get_vm_state(self, instance: Instance) -> VMState:
        raise NotImplementedError

    def machine_type(self, cores: int, worker_type: str, local_ssd: bool) -> str:  # pylint: disable=unused-argument
        return family_worker_type_cores_to_gcp_machine_type(GCP_MACHINE_FAMILY, worker_type, cores)

    def instance_config(
        self,
        machine_type: str,
        preemptible: bool,
        local_ssd_data_disk: bool,
        data_disk_size_gb: int,
        boot_disk_size_gb: int,
        job_private: bool,
        location: str,
    ):
        return LambdaSlimInstanceConfig.create(
            self.billing_manager.product_versions,
            machine_type,
            preemptible,
            job_private,
            location,
        )

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
        # if local_ssd_data_disk:
        #     assert data_disk_size_gb == 375

        # resource_rates = self.billing_manager.resource_rates

        # vm_config = create_vm_config(
        #     file_store,
        #     resource_rates,
        #     location,
        #     machine_name,
        #     machine_type,
        #     activation_token,
        #     max_idle_time_msecs,
        #     local_ssd_data_disk,
        #     data_disk_size_gb,
        #     boot_disk_size_gb,
        #     preemptible,
        #     job_private,
        #     self.project,
        #     instance_config,
        # )
        API_KEY = os.environ['LAMBDA_API_KEY']
        BASE_URL = 'https://cloud.lambdalabs.com/api/v1/'

        # Headers for authentication
        HEADERS = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }

        cores, memory_in_bytes = gcp_machine_type_to_cores_and_memory_bytes(machine_type)
        cores_mcpu = cores * 1000
        total_resources_on_instance = instance_config.quantified_resources(
            cpu_in_mcpu=cores_mcpu, memory_in_bytes=memory_in_bytes, extra_storage_in_gib=0
        )
        
        try:
            url = f'{BASE_URL}instance-operations/launch'
            payload = {
                "region_name": 'us-east-1',
                "instance_type_name": machine_type,
                "ssh_key_names": ['publickey'],
                "quantity": 1,
            }
            log.error(f'requests payload: {payload}')
            response = requests.post(url, headers=HEADERS, json=payload)
            log.info(f'created machine {machine_name} with response {response.json()}')
        except Exception:
            log.exception(f'error while creating machine {machine_name}')
        # try:
        #     params = {'requestId': str(uuid.uuid4())}
        #     await self.compute_client.post(f'/zones/{location}/instances', params=params, json=vm_config)
        #     log.info(f'created machine {machine_name}')
        # except Exception:
        #     log.exception(f'error while creating machine {machine_name}')
        return total_resources_on_instance
