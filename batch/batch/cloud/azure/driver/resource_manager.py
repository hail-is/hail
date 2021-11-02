from typing import Optional, Dict, List, Tuple

import aiohttp
import logging
import dateutil.parser

from hailtop.aiocloud import aioazure

from ....file_store import FileStore
from ....driver.resource_manager import (CloudResourceManager, VMDoesNotExist, VMState,
                                         UnknownVMState, VMStateCreating, VMStateRunning,
                                         VMStateTerminated)
from ....driver.instance import Instance
from ....instance_config import InstanceConfig, QuantifiedResource

from ..instance_config import AzureSlimInstanceConfig
from .create_instance import create_vm_config
from ..resource_utils import (azure_machine_type_to_worker_type_and_cores, azure_worker_memory_per_core_mib,
                              azure_worker_properties_to_machine_type, azure_local_ssd_size)


log = logging.getLogger('resource_manager')


def parse_azure_timestamp(timestamp: Optional[str]) -> Optional[int]:
    if timestamp is None:
        return None
    return int(dateutil.parser.isoparse(timestamp).timestamp() * 1000 + 0.5)


class AzureResourceManager(CloudResourceManager):
    def __init__(self,
                 subscription_id: str,
                 resource_group: str,
                 ssh_public_key: str,
                 arm_client: aioazure.AzureResourceManagerClient,  # BORROWED
                 compute_client: aioazure.AzureComputeClient,  # BORROWED
                 ):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.ssh_public_key = ssh_public_key
        self.arm_client = arm_client
        self.compute_client = compute_client

    async def delete_vm(self, instance: Instance):
        try:
            await self.compute_client.delete(f'/virtualMachines/{instance.name}')
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise VMDoesNotExist() from e
            raise

    async def get_vm_state(self, instance: Instance) -> VMState:
        try:
            spec = await self.compute_client.get(f'/virtualMachines/{instance.name}/instanceView')

            # https://docs.microsoft.com/en-us/azure/virtual-machines/states-billing
            for status in spec['statuses']:
                code = status['code']
                if code == 'ProvisioningState/creating':
                    return VMStateCreating(spec, instance.time_created)
                if code == 'ProvisioningState/succeeded':
                    last_start_timestamp_msecs = parse_azure_timestamp(status.get('time'))
                    assert last_start_timestamp_msecs is not None
                    return VMStateRunning(spec, last_start_timestamp_msecs)
                if code in ('ProvisioningState/failed', 'ProvisioningState/deleting', 'ProvisioningState/deleted'):
                    return VMStateTerminated(spec)

            log.exception(f'Unknown azure statuses {spec["statuses"]} for {instance}')
            return UnknownVMState(spec)

        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise VMDoesNotExist() from e
            raise

    def machine_type(self, cores: int, worker_type: str, local_ssd: bool) -> str:
        return azure_worker_properties_to_machine_type(worker_type, cores, local_ssd)

    def worker_type_and_cores(self, machine_type: str) -> Tuple[str, int]:
        return azure_machine_type_to_worker_type_and_cores(machine_type)

    def instance_config(self,
                        machine_type: str,
                        preemptible: bool,
                        local_ssd_data_disk: bool,
                        data_disk_size_gb: int,
                        boot_disk_size_gb: int,
                        job_private: bool,
                        ) -> AzureSlimInstanceConfig:
        return AzureSlimInstanceConfig(
            machine_type,
            preemptible,
            local_ssd_data_disk,
            data_disk_size_gb,
            boot_disk_size_gb,
            job_private,
        )

    def instance_config_from_dict(self, data: dict) -> AzureSlimInstanceConfig:
        return AzureSlimInstanceConfig.from_dict(data)

    async def create_vm(self,
                        file_store: FileStore,
                        resource_rates: Dict[str, float],
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
        worker_type, cores = self.worker_type_and_cores(machine_type)

        if local_ssd_data_disk:
            assert data_disk_size_gb == azure_local_ssd_size(worker_type, cores)

        vm_config = create_vm_config(file_store, resource_rates, location, machine_name,
                                     machine_type, activation_token, max_idle_time_msecs,
                                     local_ssd_data_disk, data_disk_size_gb, preemptible,
                                     job_private, self.subscription_id, self.resource_group,
                                     self.ssh_public_key, instance_config)

        memory_mib = azure_worker_memory_per_core_mib(worker_type) * cores
        memory_in_bytes = memory_mib << 20
        cores_mcpu = cores * 1000
        total_resources_on_instance = instance_config.resources(
            cpu_in_mcpu=cores_mcpu, memory_in_bytes=memory_in_bytes, extra_storage_in_gib=0
        )

        try:
            await self.arm_client.put(f'/deployments/{machine_name}-deployment', json=vm_config)
            log.info(f'created machine {machine_name}')
        except Exception:
            log.exception(f'error while creating machine {machine_name}')
        return total_resources_on_instance
