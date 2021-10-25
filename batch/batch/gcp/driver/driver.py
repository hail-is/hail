from typing import Optional

from hailtop import aiotools
from hailtop.aiocloud import aiogoogle
from hailtop.utils import periodically_call, RateLimit
from gear import Database
from gear.cloud_config import get_gcp_config

from ...driver.driver import CloudDriver, process_outstanding_events
from ...driver.instance_collection_manager import InstanceCollectionManager
from ...inst_coll_config import InstanceCollectionConfigs

from .disks import delete_orphaned_disks
from .activity_logs import process_activity_log_events_since
from .zones import get_zone, update_region_quotas, ZoneSuccessRate
from .resource_manager import GCPResourceManager


class GCPDriver(CloudDriver):
    resource_manager: GCPResourceManager
    inst_coll_manager: InstanceCollectionManager

    @staticmethod
    async def create(app, machine_name_prefix: str, inst_coll_configs: InstanceCollectionConfigs,
                     credentials_file: Optional[str] = None) -> 'GCPDriver':
        db: Database = app['db']
        task_manager = aiotools.BackgroundTaskManager()

        project = get_gcp_config().project

        compute_client = aiogoogle.GoogleComputeClient(project, credentials_file=credentials_file)

        activity_logs_client = aiogoogle.GoogleLoggingClient(
            credentials_file=credentials_file,
            # The project-wide logging quota is 60 request/m.  The event
            # loop sleeps 15s per iteration, so the max rate is 4
            # iterations/m.  Note, the event loop could make multiple
            # logging requests per iteration, so these numbers are not
            # quite comparable.  I didn't want to consume the entire quota
            # since there will be other users of the logging API (us at
            # the web console, test deployments, etc.)
            rate_limit=RateLimit(10, 60),
        )

        driver = GCPDriver(db, machine_name_prefix, task_manager, compute_client, activity_logs_client)

        resource_manager = GCPResourceManager(driver, compute_client)
        inst_coll_manager = await InstanceCollectionManager.create(app, resource_manager, machine_name_prefix, inst_coll_configs)

        driver.resource_manager = resource_manager
        driver.inst_coll_manager = inst_coll_manager

        driver.task_manager.ensure_future(periodically_call(15, driver.process_activity_logs))
        driver.task_manager.ensure_future(periodically_call(60, driver.update_region_quotas))
        driver.task_manager.ensure_future(periodically_call(60, driver.delete_orphaned_disks))

        return driver

    def __init__(self, db: Database, machine_name_prefix: str, task_manager: aiotools.BackgroundTaskManager,
                 compute_client: aiogoogle.GoogleComputeClient, activity_logs_client: aiogoogle.GoogleLoggingClient):
        self.db = db
        self.machine_name_prefix = machine_name_prefix
        self.task_manager = task_manager
        self.compute_client = compute_client
        self.activity_logs_client = activity_logs_client

        self.zone_success_rate = ZoneSuccessRate()
        self.region_info = None
        self.zones = []

        self.cloud = 'gcp'
        self.default_location = get_gcp_config().zone

    async def shutdown(self):
        try:
            self.task_manager.shutdown()
        finally:
            try:
                self.inst_coll_manager.shutdown()
            finally:
                try:
                    await self.compute_client.close()
                finally:
                    await self.activity_logs_client.close()

    def get_zone(self, cores: int, worker_local_ssd_data_disk: bool, worker_pd_ssd_data_disk_size_gb: int):
        global_live_total_cores_mcpu = self.inst_coll_manager.global_live_total_cores_mcpu
        if global_live_total_cores_mcpu // 1000 < 1_000:
            return self.default_location
        return get_zone(self.region_info, self.zone_success_rate, cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gb)

    async def process_activity_logs(self):
        async def _process_activity_log_events_since(mark):
            return await process_activity_log_events_since(self.db, self.inst_coll_manager,
                                                           self.activity_logs_client,
                                                           self.zone_success_rate, self.machine_name_prefix, mark)

        await process_outstanding_events(self.db, _process_activity_log_events_since)

    async def update_region_quotas(self):
        region_info, zones = await update_region_quotas(self.compute_client)
        self.region_info = region_info
        self.zones = zones

    async def delete_orphaned_disks(self):
        await delete_orphaned_disks(self.compute_client, self.zones, self.inst_coll_manager)
