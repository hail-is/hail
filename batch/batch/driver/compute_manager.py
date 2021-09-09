import abc
from typing import Any, Dict, Tuple

from ..worker_config import WorkerConfig

from .zone_monitor import BaseZoneMonitor
from .activity_monitor import BaseActivityMonitor
from .disk_monitor import BaseDiskMonitor

import logging
log = logging.getLogger('compute_manager')


class InstanceDoesNotExist(Exception):
    pass


class InstanceState:
    CREATING = 'Creating'
    RUNNING = 'Running'
    TERMINATED = 'Terminated'

    def __init__(self, state, full_spec, last_start_timestamp=None):
        self.state = state
        self.full_spec = full_spec
        self.last_start_timestamp = last_start_timestamp


class BaseComputeManager(abc.ABC):
    activity_monitor: BaseActivityMonitor
    disk_monitor: BaseDiskMonitor
    zone_monitor: BaseZoneMonitor
    credentials: Any

    def __init__(self, app, machine_name_prefix):
        self.app = app
        self.machine_name_prefix = machine_name_prefix

    @abc.abstractmethod
    async def delete_instance(self, instance):
        pass

    @abc.abstractmethod
    async def get_instance(self, instance) -> InstanceState:
        pass

    @abc.abstractmethod
    async def create_instance(self, *args, **kwargs):
        pass

    @abc.abstractmethod
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
        pass

    async def async_init(self):
        await self.zone_monitor.async_init()
        await self.activity_monitor.async_init()
        await self.disk_monitor.async_init()

    async def shutdown(self):
        try:
            await self.disk_monitor.shutdown()
        finally:
            try:
                await self.activity_monitor.shutdown()
            finally:
                await self.zone_monitor.shutdown()
