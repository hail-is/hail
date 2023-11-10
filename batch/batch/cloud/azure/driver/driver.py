import asyncio
import json
import logging
import os
from typing import Dict

from gear import Database
from gear.cloud_config import get_azure_config
from hailtop import aiotools
from hailtop.aiocloud import aioazure
from hailtop.utils import periodically_call

from ....driver.driver import CloudDriver
from ....driver.instance_collection import InstanceCollectionManager, JobPrivateInstanceManager, Pool
from ....inst_coll_config import InstanceCollectionConfigs
from .billing_manager import AzureBillingManager
from .regions import RegionMonitor
from .resource_manager import AzureResourceManager

log = logging.getLogger('driver')


class AzureDriver(CloudDriver):
    @staticmethod
    async def create(
        app,
        db: Database,  # BORROWED
        machine_name_prefix: str,
        namespace: str,
        inst_coll_configs: InstanceCollectionConfigs,
    ) -> 'AzureDriver':
        azure_config = get_azure_config()
        subscription_id = azure_config.subscription_id
        resource_group = azure_config.resource_group
        region = azure_config.region
        regions = [region]

        region_args = [(r,) for r in regions]
        await db.execute_many(
            '''
INSERT INTO regions (region) VALUES (%s)
ON DUPLICATE KEY UPDATE region = region;
''',
            region_args,
        )

        db_regions: Dict[str, int] = {
            record['region']: record['region_id']
            async for record in db.select_and_fetchall('SELECT region_id, region from regions')
        }
        assert max(db_regions.values()) < 64, str(db_regions)
        app['regions'] = db_regions

        with open(os.environ['HAIL_SSH_PUBLIC_KEY'], encoding='utf-8') as f:
            ssh_public_key = f.read()

        arm_client = aioazure.AzureResourceManagerClient(subscription_id, resource_group)
        compute_client = aioazure.AzureComputeClient(subscription_id, resource_group)
        resources_client = aioazure.AzureResourcesClient(subscription_id)
        network_client = aioazure.AzureNetworkClient(subscription_id, resource_group)
        pricing_client = aioazure.AzurePricingClient()

        region_monitor = await RegionMonitor.create(region)
        billing_manager = await AzureBillingManager.create(db, pricing_client, regions)
        inst_coll_manager = InstanceCollectionManager(db, machine_name_prefix, region_monitor, region, regions)
        resource_manager = AzureResourceManager(
            app, subscription_id, resource_group, ssh_public_key, arm_client, compute_client, billing_manager
        )

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

        driver = AzureDriver(
            db,
            machine_name_prefix,
            arm_client,
            compute_client,
            resources_client,
            network_client,
            pricing_client,
            subscription_id,
            resource_group,
            namespace,
            region_monitor,
            inst_coll_manager,
            jpim,
            billing_manager,
            task_manager,
        )

        task_manager.ensure_future(periodically_call(60, driver.delete_orphaned_nics))
        task_manager.ensure_future(periodically_call(60, driver.delete_orphaned_public_ips))
        task_manager.ensure_future(periodically_call(60, driver.delete_completed_deployments))
        task_manager.ensure_future(periodically_call(300, billing_manager.refresh_resources_from_retail_prices))

        return driver

    def __init__(
        self,
        db: Database,
        machine_name_prefix: str,
        arm_client: aioazure.AzureResourceManagerClient,
        compute_client: aioazure.AzureComputeClient,
        resources_client: aioazure.AzureResourcesClient,
        network_client: aioazure.AzureNetworkClient,
        pricing_client: aioazure.AzurePricingClient,
        subscription_id: str,
        resource_group: str,
        namespace: str,
        region_monitor: RegionMonitor,
        inst_coll_manager: InstanceCollectionManager,
        job_private_inst_manager: JobPrivateInstanceManager,
        billing_manager: AzureBillingManager,
        task_manager: aiotools.BackgroundTaskManager,
    ):
        self.db = db
        self.machine_name_prefix = machine_name_prefix
        self.arm_client = arm_client
        self.compute_client = compute_client
        self.resources_client = resources_client
        self.network_client = network_client
        self.pricing_client = pricing_client
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.namespace = namespace
        self.region_monitor = region_monitor
        self.job_private_inst_manager = job_private_inst_manager
        self._billing_manager = billing_manager
        self._inst_coll_manager = inst_coll_manager
        self._task_manager = task_manager

    @property
    def billing_manager(self) -> AzureBillingManager:
        return self._billing_manager

    @property
    def inst_coll_manager(self) -> InstanceCollectionManager:
        return self._inst_coll_manager

    async def shutdown(self) -> None:
        try:
            await self._task_manager.shutdown_and_wait()
        finally:
            try:
                await self.arm_client.close()
            finally:
                try:
                    await self.compute_client.close()
                finally:
                    try:
                        await self.resources_client.close()
                    finally:
                        try:
                            await self.network_client.close()
                        finally:
                            await self.pricing_client.close()

    def _resource_is_orphaned(self, resource_name: str) -> bool:
        instance_name = resource_name.rsplit('-', maxsplit=1)[0]
        return self.inst_coll_manager.get_instance(instance_name) is None

    async def delete_orphaned_nics(self) -> None:
        log.info('deleting orphaned nics')
        async for nic_name in self.resources_client.list_nic_names(self.machine_name_prefix):
            if self._resource_is_orphaned(nic_name):
                try:
                    log.info(f'deleting orphaned nic {nic_name}')
                    await self.network_client.delete_nic(nic_name, ignore_not_found=True)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception(f'while deleting orphaned nic {nic_name}')

    async def delete_orphaned_public_ips(self) -> None:
        log.info('deleting orphaned public ips')
        async for public_ip_name in self.resources_client.list_public_ip_names(self.machine_name_prefix):
            if self._resource_is_orphaned(public_ip_name):
                try:
                    log.info(f'deleting orphaned public ip {public_ip_name}')
                    await self.network_client.delete_public_ip(public_ip_name, ignore_not_found=True)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception(f'while deleting orphaned public ip {public_ip_name}')

    async def delete_completed_deployments(self) -> None:
        log.info('deleting completed deployments')
        # https://docs.microsoft.com/en-us/rest/api/resources/deployments/list-by-resource-group#provisioningstate
        for state in ('Succeeded', 'Failed', 'Canceled'):
            async for deployment in await self.arm_client.list_deployments(filter=f"provisioningState eq '{state}'"):
                deployment_name = deployment['name']
                if deployment_name.startswith(self.machine_name_prefix):
                    try:
                        deployment.pop('parameters', None)
                        log.info(f'deleting deployment {deployment_name} {json.dumps(deployment)}')
                        await self.arm_client.delete(f'/deployments/{deployment_name}')
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        log.exception(f'while deleting completed deployment {deployment_name} {deployment}')

    def get_quotas(self):
        raise NotImplementedError
