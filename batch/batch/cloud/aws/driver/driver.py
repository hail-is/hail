import asyncio
import logging
from typing import Dict

from gear import Database
from gear.cloud_config import get_aws_config
from hailtop import aiotools

from ....driver.driver import CloudDriver
from ....driver.instance_collection import InstanceCollectionManager, JobPrivateInstanceManager, Pool
from ....inst_coll_config import InstanceCollectionConfigs
from .billing_manager import AWSBillingManager
from .location import AWSLocationMonitor
from .resource_manager import AWSResourceManager

log = logging.getLogger('driver')


class AWSDriver(CloudDriver):
    @staticmethod
    async def create(
        app,
        db: Database,  # BORROWED
        machine_name_prefix: str,
        namespace: str,
        inst_coll_configs: InstanceCollectionConfigs,
    ) -> 'AWSDriver':
        aws_config = get_aws_config()
        region = aws_config.region
        regions = list(aws_config.regions)

        region_args = [(region,) for region in regions]
        await db.execute_many(
            """
INSERT INTO regions (region) VALUES (%s)
ON DUPLICATE KEY UPDATE region = region;
""",
            region_args,
        )

        db_regions: Dict[str, int] = {
            record['region']: record['region_id']
            async for record in db.select_and_fetchall('SELECT region_id, region from regions')
        }
        assert max(db_regions.values()) < 64, str(db_regions)
        app['regions'] = db_regions

        location_monitor = AWSLocationMonitor(region)
        billing_manager = await AWSBillingManager.create(db)
        inst_coll_manager = InstanceCollectionManager(db, machine_name_prefix, location_monitor, region, regions)
        resource_manager = AWSResourceManager()

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
            for config in inst_coll_configs.name_pool_config.values()
        ]

        jpim, *_ = await asyncio.gather(
            JobPrivateInstanceManager.create(
                app,
                db,
                inst_coll_manager,
                resource_manager,
                machine_name_prefix,
                inst_coll_configs.jpim_config,
                task_manager,
            ),
            *create_pools_coros,
        )

        assert isinstance(jpim, JobPrivateInstanceManager)
        driver = AWSDriver(
            db,
            machine_name_prefix,
            namespace,
            location_monitor,
            inst_coll_manager,
            jpim,
            billing_manager,
            task_manager,
        )

        # TODO: add periodic driver tasks

        return driver

    def __init__(
        self,
        db: Database,
        machine_name_prefix: str,
        namespace: str,
        location_monitor: AWSLocationMonitor,
        inst_coll_manager: InstanceCollectionManager,
        job_private_inst_manager: JobPrivateInstanceManager,
        billing_manager: AWSBillingManager,
        task_manager: aiotools.BackgroundTaskManager,
    ):
        self.db = db
        self.machine_name_prefix = machine_name_prefix
        self.namespace = namespace
        self.location_monitor = location_monitor
        self.job_private_inst_manager = job_private_inst_manager
        self._billing_manager = billing_manager
        self._inst_coll_manager = inst_coll_manager
        self._task_manager = task_manager

    @property
    def billing_manager(self) -> AWSBillingManager:
        return self._billing_manager

    @property
    def inst_coll_manager(self) -> InstanceCollectionManager:
        return self._inst_coll_manager

    async def shutdown(self) -> None:
        try:
            await self._task_manager.shutdown_and_wait()
        finally:
            await self._billing_manager.close()

    def get_quotas(self):
        return {}
