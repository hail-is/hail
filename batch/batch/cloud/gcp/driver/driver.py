import asyncio
from hailtop import aiotools
from hailtop.aiocloud import aiogoogle
from hailtop.utils import periodically_call, RateLimit
from gear import Database
from gear.cloud_config import get_gcp_config

from ....driver.driver import CloudDriver, process_outstanding_events
from ....driver.instance_collection import Pool, JobPrivateInstanceManager, InstanceCollectionManager
from ....inst_coll_config import InstanceCollectionConfigs


from .disks import delete_orphaned_disks
from .activity_logs import process_activity_log_events_since
from .zones import ZoneMonitor
from .resource_manager import GCPResourceManager


class GCPDriver(CloudDriver):
    @staticmethod
    async def create(app,
                     db: Database,  # BORROWED
                     machine_name_prefix: str,
                     namespace: str,
                     inst_coll_configs: InstanceCollectionConfigs,
                     credentials_file: str,
                     task_manager: aiotools.BackgroundTaskManager,  # BORROWED
                     ) -> 'GCPDriver':
        gcp_config = get_gcp_config()
        project = gcp_config.project
        zone = gcp_config.zone
        regions = gcp_config.regions

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

        zone_monitor = await ZoneMonitor.create(compute_client, regions, zone)
        inst_coll_manager = InstanceCollectionManager(db, machine_name_prefix, zone_monitor)
        resource_manager = GCPResourceManager(project, compute_client)

        create_pools_coros = [
            Pool.create(app,
                        db,
                        inst_coll_manager,
                        resource_manager,
                        machine_name_prefix,
                        config,
                        app['async_worker_pool'],
                        task_manager)
            for pool_name, config in inst_coll_configs.name_pool_config.items()
        ]

        jpim, *_ = await asyncio.gather(
            JobPrivateInstanceManager.create(
                app, db, inst_coll_manager, resource_manager, machine_name_prefix, inst_coll_configs.jpim_config, task_manager),
            *create_pools_coros)

        driver = GCPDriver(db,
                           machine_name_prefix,
                           compute_client,
                           activity_logs_client,
                           project,
                           namespace,
                           zone_monitor,
                           inst_coll_manager,
                           jpim)

        task_manager.ensure_future(periodically_call(15, driver.process_activity_logs))
        task_manager.ensure_future(periodically_call(60, zone_monitor.update_region_quotas))
        task_manager.ensure_future(periodically_call(60, driver.delete_orphaned_disks))

        return driver

    def __init__(self,
                 db: Database,
                 machine_name_prefix: str,
                 compute_client: aiogoogle.GoogleComputeClient,
                 activity_logs_client: aiogoogle.GoogleLoggingClient,
                 project: str,
                 namespace: str,
                 zone_monitor: ZoneMonitor,
                 inst_coll_manager: InstanceCollectionManager,
                 job_private_inst_manager: JobPrivateInstanceManager):
        self.db = db
        self.machine_name_prefix = machine_name_prefix
        self.compute_client = compute_client
        self.activity_logs_client = activity_logs_client
        self.project = project
        self.namespace = namespace
        self.zone_monitor = zone_monitor
        self.inst_coll_manager = inst_coll_manager
        self.job_private_inst_manager = job_private_inst_manager

    async def shutdown(self) -> None:
        try:
            await self.compute_client.close()
        finally:
            await self.activity_logs_client.close()

    async def process_activity_logs(self) -> None:
        async def _process_activity_log_events_since(mark):
            return await process_activity_log_events_since(self.db,
                                                           self.inst_coll_manager,
                                                           self.activity_logs_client,
                                                           self.zone_monitor.zone_success_rate,
                                                           self.machine_name_prefix,
                                                           self.project,
                                                           mark)

        await process_outstanding_events(self.db, _process_activity_log_events_since)

    async def delete_orphaned_disks(self) -> None:
        await delete_orphaned_disks(self.compute_client, self.zone_monitor.zones, self.inst_coll_manager, self.namespace)
