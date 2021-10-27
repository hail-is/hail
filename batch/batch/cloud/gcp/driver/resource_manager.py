from typing import Optional, TYPE_CHECKING

import aiohttp
import logging
import uuid
import dateutil.parser

from hailtop.aiocloud import aiogoogle

from ....driver.resource_manager import CloudResourceManager, VMDoesNotExist, VMState
from ....driver.instance import Instance

from ..instance_config import GCPInstanceConfig
from .create_instance import create_instance_config
from ..resource_utils import gcp_machine_type_to_worker_type_cores

if TYPE_CHECKING:
    from .driver import GCPDriver  # pylint: disable=cyclic-import
    from ....driver.instance_collection import InstanceCollection  # pylint: disable=cyclic-import


log = logging.getLogger('resource_manager')


def parse_gcp_timestamp(timestamp: Optional[str]) -> Optional[int]:
    if timestamp is None:
        return None
    return int(dateutil.parser.isoparse(timestamp).timestamp() * 1000 + 0.5)


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

            last_start_timestamp_msecs = parse_gcp_timestamp(spec.get('lastStartTimestamp'))
            last_stop_timestamp_msecs = parse_gcp_timestamp(spec.get('lastStopTimestamp'))

            if state in ('PROVISIONING', 'STAGING'):
                vm_state = VMState(VMState.CREATING, spec, instance.time_created)
            elif state == 'RUNNING':
                vm_state = VMState(VMState.RUNNING, spec, last_start_timestamp_msecs)
            elif state in ('STOPPING', 'TERMINATED'):
                vm_state = VMState(VMState.TERMINATED, spec, last_stop_timestamp_msecs)
            else:
                log.exception(f'Unknown gce state {state} for {instance}')
                vm_state = VMState(VMState.UNKNOWN, spec, None)

            return vm_state

        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise VMDoesNotExist() from e
            raise

    async def create_vm(self,
                        app,
                        inst_coll: 'InstanceCollection',
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
                        ) -> Optional[Instance]:
        if machine_type is None:
            assert worker_type and cores
            machine_type = f'n1-{worker_type}-{cores}'
        else:
            _, cores = gcp_machine_type_to_worker_type_cores(machine_type)

        zone = location
        if zone is None:
            assert cores is not None, (cores, zone)
            zone = self.driver.get_zone(cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb)
            if zone is None:
                return None

        instance_config = create_instance_config(app, zone, machine_name, machine_type, activation_token, max_idle_time_msecs,
                                                 worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb, boot_disk_size_gb,
                                                 preemptible, job_private, self.project)

        if instance_config is None:
            return None

        instance = await Instance.create(
            app=app,
            inst_coll=inst_coll,
            name=machine_name,
            activation_token=activation_token,
            instance_config=instance_config
        )

        try:
            params = {'requestId': str(uuid.uuid4())}
            await self.compute_client.post(f'/zones/{zone}/instances', params=params, json=instance_config.vm_config)
            log.info(f'created machine {machine_name}')
        except Exception:
            log.exception(f'error while creating machine {machine_name}')
        finally:
            return instance  # pylint: disable=lost-exception
