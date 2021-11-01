from typing import Optional, Dict, List, Any, Tuple

import aiohttp
import logging
import uuid
import dateutil.parser

from hailtop.aiocloud import aiogoogle

from ....file_store import FileStore
from ....driver.resource_manager import (CloudResourceManager, VMDoesNotExist, VMState,
                                         UnknownVMState, VMStateCreating, VMStateRunning,
                                         VMStateTerminated)
from ....driver.instance import Instance
from ....instance_config import InstanceConfig

from ..instance_config import GCPSlimInstanceConfig
from .create_instance import create_vm_config
from ..resource_utils import (gcp_machine_type_to_worker_type_cores, gcp_worker_memory_per_core_mib,
                              family_worker_type_cores_to_gcp_machine_type, GCP_MACHINE_FAMILY)


log = logging.getLogger('resource_manager')


def parse_gcp_timestamp(timestamp: Optional[str]) -> Optional[int]:
    if timestamp is None:
        return None
    return int(dateutil.parser.isoparse(timestamp).timestamp() * 1000 + 0.5)


class GCPResourceManager(CloudResourceManager):
    def __init__(self,
                 project: str,
                 compute_client: aiogoogle.GoogleComputeClient,  # BORROWED
                 ):
        self.compute_client = compute_client
        self.project = project

    async def delete_vm(self, instance: Instance):
        try:
            await self.compute_client.delete(f'/zones/{instance.location}/instances/{instance.name}')
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise VMDoesNotExist() from e
            raise

    async def get_vm_state(self, instance: Instance) -> VMState:
        try:
            spec = await self.compute_client.get(f'/zones/{instance.location}/instances/{instance.name}')
            state = spec['status']  # PROVISIONING, STAGING, RUNNING, STOPPING, TERMINATED

            if state in ('PROVISIONING', 'STAGING'):
                return VMStateCreating(spec, instance.time_created)
            if state == 'RUNNING':
                last_start_timestamp_msecs = parse_gcp_timestamp(spec.get('lastStartTimestamp'))
                assert last_start_timestamp_msecs is not None
                return VMStateRunning(spec, last_start_timestamp_msecs)
            if state in ('STOPPING', 'TERMINATED'):
                last_stop_timestamp_msecs = parse_gcp_timestamp(spec.get('lastStopTimestamp'))
                assert last_stop_timestamp_msecs is not None
                return VMStateTerminated(spec, last_stop_timestamp_msecs)
            log.exception(f'Unknown gce state {state} for {instance}')
            return UnknownVMState(spec)

        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise VMDoesNotExist() from e
            raise

    def machine_type(self, cores: int, worker_type: str) -> str:
        return family_worker_type_cores_to_gcp_machine_type(
            GCP_MACHINE_FAMILY, worker_type, cores)

    def worker_type_cores(self, machine_type: str) -> Tuple[str, int]:
        return gcp_machine_type_to_worker_type_cores(machine_type)

    def instance_config(self,
                        machine_type: str,
                        preemptible: bool,
                        local_ssd_data_disk: bool,
                        data_disk_size_gb: int,
                        boot_disk_size_gb: int,
                        job_private: bool,
                        ) -> GCPSlimInstanceConfig:
        return GCPSlimInstanceConfig(
            machine_type,
            preemptible,
            local_ssd_data_disk,
            data_disk_size_gb,
            boot_disk_size_gb,
            job_private,
        )

    def instance_config_from_dict(self, data: dict) -> GCPSlimInstanceConfig:
        return GCPSlimInstanceConfig.from_dict(data)

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
                        ) -> List[Dict[str, Any]]:
        if local_ssd_data_disk:
            assert data_disk_size_gb == 375

        worker_type, cores = self.worker_type_cores(machine_type)
        vm_config = create_vm_config(file_store, resource_rates, location, machine_name,
                                     machine_type, activation_token, max_idle_time_msecs,
                                     local_ssd_data_disk, data_disk_size_gb, boot_disk_size_gb,
                                     preemptible, job_private, self.project, instance_config)

        memory_mib = gcp_worker_memory_per_core_mib(worker_type) * cores
        memory_in_bytes = memory_mib << 20
        cores_mcpu = cores * 1000
        total_resources_on_instance = instance_config.resources(
            cpu_in_mcpu=cores_mcpu, memory_in_bytes=memory_in_bytes, extra_storage_in_gib=0
        )

        try:
            params = {'requestId': str(uuid.uuid4())}
            await self.compute_client.post(f'/zones/{location}/instances', params=params, json=vm_config)
            log.info(f'created machine {machine_name}')
        except Exception:
            log.exception(f'error while creating machine {machine_name}')
        return total_resources_on_instance
