from typing import Optional, TYPE_CHECKING

import aiohttp
import logging
import uuid

from hailtop import aiotools
from hailtop.aiocloud import aiogoogle
from hailtop.utils import RateLimit, periodically_call
from gear import Database

from ...batch_configuration import PROJECT, GCP_ZONE
from ...driver.resource_manager import CloudResourceManager, VMDoesNotExist, VMState, process_outstanding_events

from ..instance_config import GCPInstanceConfig
from .create_instance import create_instance_config
from .disks import delete_orphaned_disks
from .activity_logs import process_activity_log_events_since
from .zones import get_zone, update_region_quotas, ZoneSuccessRate

if TYPE_CHECKING:
    from ...driver.instance_collection_manager import InstanceCollectionManager  # pylint: disable=cyclic-import

log = logging.getLogger('resource_manager')


class GCPResourceManager(CloudResourceManager):
    def __init__(self, app, machine_name_prefix, credentials: Optional[aiogoogle.GoogleCredentials] = None):
        self.app = app
        self.db: Database = app['db']
        self.machine_name_prefix = machine_name_prefix

        if credentials is None:
            credentials = aiogoogle.GoogleCredentials.from_file('/gsa-key/key.json')

        self.compute_client = aiogoogle.GoogleComputeClient(PROJECT, credentials=credentials)

        self.activity_logs_client = aiogoogle.GoogleLoggingClient(
            credentials=credentials,
            # The project-wide logging quota is 60 request/m.  The event
            # loop sleeps 15s per iteration, so the max rate is 4
            # iterations/m.  Note, the event loop could make multiple
            # logging requests per iteration, so these numbers are not
            # quite comparable.  I didn't want to consume the entire quota
            # since there will be other users of the logging API (us at
            # the web console, test deployments, etc.)
            rate_limit=RateLimit(10, 60),
        )

        self.zone_success_rate = ZoneSuccessRate()
        self.region_info = None
        self.zones = []

        self.cloud = 'gcp'
        self.default_location = GCP_ZONE

        self.task_manager = aiotools.BackgroundTaskManager()

    async def async_init(self):
        self.task_manager.ensure_future(periodically_call(15, self.process_activity_logs))
        self.task_manager.ensure_future(periodically_call(60, self.update_region_quotas))
        self.task_manager.ensure_future(periodically_call(60, self.delete_orphaned_disks))

    def shutdown(self):
        try:
            self.task_manager.shutdown()
        finally:
            try:
                await self.compute_client.close()
            finally:
                await self.activity_logs_client.close()

    async def delete_vm(self, instance):
        instance_config = instance.instance_config
        assert isinstance(instance_config, GCPInstanceConfig)
        try:
            await self.compute_client.delete(f'/zones/{instance_config.zone}/instances/{instance_config.name}')
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise VMDoesNotExist() from e
            raise

    async def get_vm_state(self, instance) -> VMState:
        instance_config = instance.instance_config
        assert isinstance(instance_config, GCPInstanceConfig)
        try:
            spec = await self.compute_client.get(f'/zones/{instance_config.zone}/instances/{instance_config.name}')
            state = spec['status']  # PROVISIONING, STAGING, RUNNING, STOPPING, TERMINATED

            if state in 'PROVISIONING':
                state = VMState.CREATING
            elif state in ('STAGING', 'RUNNING'):
                state = VMState.RUNNING
            elif state in ('STOPPING', 'TERMINATED'):
                state = VMState.TERMINATED
            else:
                log.exception(f'Unknown gce stat {state} for {instance}')
                state = None

            return VMState(state, spec, spec.get('lastStartTimestamp'))

        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise VMDoesNotExist() from e
            raise

    def prepare_vm(self,
                   app,
                   machine_name,
                   activation_token,
                   max_idle_time_msecs,
                   worker_local_ssd_data_disk,
                   worker_pd_ssd_data_disk_size_gb,
                   boot_disk_size_gb,
                   preemptible,
                   job_private,
                   machine_type=None,
                   worker_type=None,
                   cores=None,
                   location=None,
                   ) -> Optional[GCPInstanceConfig]:
        assert machine_type or (worker_type and cores)

        if machine_type is None:
            machine_type = f'n1-{worker_type}-{cores}'

        zone = location
        if zone is None:
            # FIXME: We can get this info from the region quotas in the future
            inst_coll_manager: 'InstanceCollectionManager' = self.app['inst_coll_manager']
            global_live_total_cores_mcpu = inst_coll_manager.global_live_total_cores_mcpu
            if global_live_total_cores_mcpu // 1000 < 1_000:
                zone = self.default_location
            else:
                zone = get_zone(self.region_info, self.zone_success_rate, cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb)
                if zone is None:
                    return None

        return create_instance_config(app, zone, machine_name, machine_type, activation_token, max_idle_time_msecs,
                                      worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb, boot_disk_size_gb,
                                      preemptible, job_private)

    async def create_vm(self, instance_config: GCPInstanceConfig):
        machine_name = instance_config.name
        zone = instance_config.zone
        params = {'requestId': str(uuid.uuid4())}
        await self.compute_client.post(f'/zones/{zone}/instances', params=params, json=instance_config.vm_config)
        log.info(f'created machine {machine_name}')

    async def process_activity_logs(self):
        async def _process_activity_log_events_since(mark):
            return await process_activity_log_events_since(self.db, self.app['inst_coll_manager'],
                                                           self.activity_logs_client,
                                                           self.zone_success_rate, self.machine_name_prefix, mark)

        await process_outstanding_events(self.db, _process_activity_log_events_since)

    async def update_region_quotas(self):
        await update_region_quotas(self.compute_client)

    async def delete_orphaned_disks(self):
        await delete_orphaned_disks(self.compute_client, self.zones, self.app['inst_coll_manager'])
