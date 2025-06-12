import asyncio
from typing import Dict

from gear import Database
from hailtop import aiotools
from hailtop.aiocloud import aiogoogle
from hailtop.utils import RateLimit, periodically_call

from ....driver.driver import CloudDriver, process_outstanding_events
from ....driver.instance_collection import InstanceCollectionManager, JobPrivateInstanceManager, Pool
from ....inst_coll_config import InstanceCollectionConfigs
from .activity_logs import process_activity_log_events_since
from .billing_manager import GCPBillingManager
from .disks import delete_orphaned_disks
from batch.batch.cloud.lambda_.driver.resource_manager import GCPResourceManager
from .zones import ZoneMonitor


class LambdaDriver(CloudDriver):
    @staticmethod
    async def create(
        app,
        db: Database,  # BORROWED
        machine_name_prefix: str,
        namespace: str,
        inst_coll_configs: InstanceCollectionConfigs,
        cloud: str,
    ) -> 'LambdaDriver':
        lambda_config = get_lambda_config()

        # put lambda configuration parameters that are needed here

        project = gcp_config.project
        zone = gcp_config.zone
        region = gcp_config.region
        regions = gcp_config.regions

        region_args = [(region, cloud) for region in regions]
        await db.execute_many(
            """
INSERT INTO regions (region, cloud) VALUES (%s, %s)
ON DUPLICATE KEY UPDATE region = region;
""",
            region_args,
        )

        db_regions: Dict[str, int] = {
            record['region']: record['region_id']
            async for record in db.select_and_fetchall('SELECT region_id, region from regions WHERE cloud = %s;', (cloud,))
        }
        assert max(db_regions.values()) < 64, str(db_regions)
        app['regions'] = db_regions

        compute_client = ...

        activity_logs_client = ...

        zone_monitor = ...
        billing_manager = ...
        inst_coll_manager = InstanceCollectionManager(db, machine_name_prefix, zone_monitor, region, regions)
        resource_manager = GCPResourceManager(project, compute_client, billing_manager)

        task_manager = aiotools.BackgroundTaskManager()

        create_pools_coros = [
            Pool.create(
                app,
                db,
                inst_coll_manager,
                resource_manager,
                machine_name_prefix,
                config,
                app['async_worker_pool'],
                task_manager,
            )
            for config in inst_coll_configs.name_pool_config[cloud].values()
        ]

        jpim, *_ = await asyncio.gather(
            JobPrivateInstanceManager.create(
                app,
                db,
                inst_coll_manager,
                resource_manager,
                machine_name_prefix,
                inst_coll_configs.jpim_config[cloud],
                task_manager,
            ),
            *create_pools_coros,
        )

        assert isinstance(jpim, JobPrivateInstanceManager)
        driver = LambdaDriver(
            db,
            machine_name_prefix,
            compute_client,
            activity_logs_client,
            project,
            namespace,
            zone_monitor,
            inst_coll_manager,
            jpim,
            billing_manager,
            task_manager,
        )

        task_manager.ensure_future(periodically_call(15, driver.process_activity_logs))
        task_manager.ensure_future(periodically_call(60, zone_monitor.update_region_quotas))
        task_manager.ensure_future(periodically_call(60, driver.delete_orphaned_disks))
        task_manager.ensure_future(periodically_call(300, billing_manager.refresh_resources_from_retail_prices))

        return driver

    def __init__(
        self,
        db: Database,
        machine_name_prefix: str,
        compute_client: aiogoogle.GoogleComputeClient,
        activity_logs_client: aiogoogle.GoogleLoggingClient,
        project: str,
        namespace: str,
        zone_monitor: ZoneMonitor,
        inst_coll_manager: InstanceCollectionManager,
        job_private_inst_manager: JobPrivateInstanceManager,
        billing_manager: GCPBillingManager,
        task_manager: aiotools.BackgroundTaskManager,
    ):
        self.db = db
        self.machine_name_prefix = machine_name_prefix
        self.compute_client = compute_client
        self.activity_logs_client = activity_logs_client
        self.project = project
        self.namespace = namespace
        self.zone_monitor = zone_monitor
        self.job_private_inst_manager = job_private_inst_manager
        self._billing_manager = billing_manager
        self._inst_coll_manager = inst_coll_manager
        self._task_manager = task_manager

    @property
    def billing_manager(self) -> GCPBillingManager:
        return self._billing_manager

    @property
    def inst_coll_manager(self) -> InstanceCollectionManager:
        return self._inst_coll_manager

    async def shutdown(self) -> None:
        try:
            await self._task_manager.shutdown_and_wait()
        finally:
            try:
                await self.compute_client.close()
            finally:
                try:
                    await self.activity_logs_client.close()
                finally:
                    await self._billing_manager.close()

    async def process_activity_logs(self) -> None:
        async def _process_activity_log_events_since(mark):
            return await process_activity_log_events_since(
                self.db,
                self.inst_coll_manager,
                self.activity_logs_client,
                self.zone_monitor.zone_success_rate,
                self.machine_name_prefix,
                self.project,
                mark,
            )

        await process_outstanding_events(self.db, _process_activity_log_events_since)

    async def delete_orphaned_disks(self) -> None:
        await delete_orphaned_disks(
            self.compute_client, self.zone_monitor.zones, self.inst_coll_manager, self.namespace
        )

    def get_quotas(self):
        return self.zone_monitor.region_quotas
