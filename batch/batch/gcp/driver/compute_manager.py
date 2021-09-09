from typing import Any, Dict, Tuple
import aiohttp
import logging
import uuid

from hailtop import aiogoogle

from ...batch_configuration import PROJECT
from ...worker_config import WorkerConfig
from ...driver.compute_manager import BaseComputeManager, InstanceDoesNotExist, InstanceState

from .create_instance import create_instance_config
from .zone_monitor import ZoneMonitor
from .activity_monitor import ActivityMonitor
from .disk_monitor import DiskMonitor

log = logging.getLogger('compute_manager')


class ComputeManager(BaseComputeManager):
    def __init__(self, app, machine_name_prefix: str):
        super().__init__(app, machine_name_prefix)
        self.credentials = aiogoogle.Credentials.from_file('/gsa-key/key.json')
        self.compute_client = aiogoogle.ComputeClient(PROJECT, credentials=self.credentials)

        self.zone_monitor = ZoneMonitor(self)
        self.disk_monitor = DiskMonitor(self)
        self.activity_monitor = ActivityMonitor(self)

    async def delete_instance(self, instance):
        try:
            await self.compute_client.delete(f'/zones/{instance.zone}/instances/{instance.name}')
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise InstanceDoesNotExist() from e
            raise

    async def get_instance(self, instance) -> InstanceState:
        try:
            spec = await self.compute_client.get(f'/zones/{instance.zone}/instances/{instance.name}')
            state = spec['status']  # PROVISIONING, STAGING, RUNNING, STOPPING, TERMINATED

            if state in 'PROVISIONING':
                state = InstanceState.CREATING
            elif state in ('STAGING', 'RUNNING'):
                state = InstanceState.RUNNING
            elif state in ('STOPPING', 'TERMINATED'):
                state = InstanceState.TERMINATED
            else:
                log.exception(f'Unknown gce stat {state} for {instance}')
                state = None

            return InstanceState(state, spec, spec.get('lastStartTimestamp'))

        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise InstanceDoesNotExist() from e
            raise

    async def create_instance(self, machine_name, zone, config):
        params = {'requestId': str(uuid.uuid4())}
        await self.compute_client.post(f'/zones/{zone}/instances', params=params, json=config)
        log.info(f'created machine {machine_name}')

    def create_instance_config(
            self,
            app,
            zone,
            machine_name,
            machine_type,
            activation_token,
            max_idle_time_msecs,
            worker_local_ssd_data_disk,
            worker_pd_ssd_data_disk_size_gb,
            boot_disk_size_gb,
            preemptible,
            job_private,
    ) -> Tuple[Dict[str, Any], WorkerConfig]:
        return create_instance_config(app, zone, machine_name, machine_type, activation_token,
                                      max_idle_time_msecs, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb,
                                      boot_disk_size_gb, preemptible, job_private)

    async def shutdown(self):
        try:
            await super().shutdown()
        finally:
            await self.compute_client.close()
