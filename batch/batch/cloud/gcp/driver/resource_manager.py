from typing import Optional, TYPE_CHECKING

import aiohttp
import logging
import uuid
import dateutil.parser

from hailtop.aiocloud import aiogoogle

from ....driver.resource_manager import CloudResourceManager, VMDoesNotExist, VMState

from ..instance_config import GCPInstanceConfig
from .create_instance import create_instance_config


if TYPE_CHECKING:
    from .driver import GCPDriver  # pylint: disable=cyclic-import
    from ....driver.instance import Instance  # pylint: disable=cyclic-import


log = logging.getLogger('resource_manager')


def parse_gcp_timestamp(timestamp: Optional[str]) -> Optional[float]:
    if timestamp is None:
        return None
    return dateutil.parser.isoparse(timestamp).timestamp() * 1000


class GCPResourceManager(CloudResourceManager):
    def __init__(self, driver: 'GCPDriver', compute_client: aiogoogle.GoogleComputeClient, default_location: str):
        self.driver = driver
        self.compute_client = compute_client
        self.default_location = default_location
        self.project = driver.project

    async def delete_vm(self, instance: 'Instance'):
        instance_config = instance.instance_config
        assert isinstance(instance_config, GCPInstanceConfig)
        try:
            await self.compute_client.delete(f'/zones/{instance_config.zone}/instances/{instance_config.name}')
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise VMDoesNotExist() from e
            raise

    async def get_vm_state(self, instance: 'Instance') -> VMState:
        instance_config = instance.instance_config
        assert isinstance(instance_config, GCPInstanceConfig)
        try:
            spec = await self.compute_client.get(f'/zones/{instance_config.zone}/instances/{instance_config.name}')
            state = spec['status']  # PROVISIONING, STAGING, RUNNING, STOPPING, TERMINATED

            if state in ('PROVISIONING', 'STAGING'):
                vm_state = VMState(VMState.CREATING, spec, instance.time_created)
            elif state == 'RUNNING':
                last_start_timestamp_msecs = parse_gcp_timestamp(spec.get('lastStartTimestamp'))
                vm_state = VMState(VMState.RUNNING, spec, last_start_timestamp_msecs)
            elif state in ('STOPPING', 'TERMINATED'):
                last_stop_timestamp_msecs = parse_gcp_timestamp(spec.get('lastStopTimestamp'))
                vm_state = VMState(VMState.TERMINATED, spec, last_stop_timestamp_msecs)
            else:
                log.exception(f'Unknown gce state {state} for {instance}')
                vm_state = VMState(VMState.UNKNOWN, spec, None)

            return vm_state

        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise VMDoesNotExist() from e
            raise

    def prepare_vm(self,
                   app,
                   machine_name: str,
                   activation_token: str,
                   max_idle_time_msecs: int,
                   worker_local_ssd_data_disk: bool,
                   worker_pd_ssd_data_disk_size_gb: int,
                   boot_disk_size_gb: int,
                   preemptible: bool,
                   job_private: bool,
                   machine_type: Optional[str] = None,
                   worker_type: Optional[str] = None,
                   cores: Optional[int] = None,
                   location: Optional[str] = None,
                   ) -> Optional[GCPInstanceConfig]:
        assert machine_type or (worker_type and cores)

        if machine_type is None:
            machine_type = f'n1-{worker_type}-{cores}'

        zone = location
        if zone is None:
            zone = self.driver.get_zone(cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb)
            if zone is None:
                return None

        return create_instance_config(app, zone, machine_name, machine_type, activation_token, max_idle_time_msecs,
                                      worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb, boot_disk_size_gb,
                                      preemptible, job_private, self.project)

    async def create_vm(self, instance_config: GCPInstanceConfig):
        machine_name = instance_config.name
        zone = instance_config.zone
        params = {'requestId': str(uuid.uuid4())}
        await self.compute_client.post(f'/zones/{zone}/instances', params=params, json=instance_config.vm_config)
        log.info(f'created machine {machine_name}')
