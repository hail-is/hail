import asyncio
import logging
import os

from hailtop import aiotools
from hailtop.utils import periodically_call
from hailtop.aiocloud import aioazure
from gear import Database
from gear.cloud_config import get_azure_config

from ....driver.driver import CloudDriver
from ....driver.instance_collection import Pool, JobPrivateInstanceManager, InstanceCollectionManager
from ....inst_coll_config import InstanceCollectionConfigs

from .resource_manager import AzureResourceManager
from .regions import RegionMonitor


log = logging.getLogger('driver')


class AzureDriver(CloudDriver):
    @staticmethod
    async def create(app,
                     db: Database,  # BORROWED
                     machine_name_prefix: str,
                     namespace: str,
                     inst_coll_configs: InstanceCollectionConfigs,
                     credentials_file: str,
                     task_manager: aiotools.BackgroundTaskManager,  # BORROWED
                     ) -> 'AzureDriver':
        azure_config = get_azure_config()
        subscription_id = azure_config.subscription_id
        resource_group = azure_config.resource_group
        region = azure_config.region

        with open(os.environ['HAIL_SSH_PUBLIC_KEY']) as f:
            ssh_public_key = f.read()

        arm_client = aioazure.AzureResourceManagerClient(subscription_id, resource_group, credentials_file=credentials_file)
        compute_client = aioazure.AzureComputeClient(subscription_id, resource_group, credentials_file=credentials_file)
        resources_client = aioazure.AzureResourcesClient(subscription_id, credentials_file=credentials_file)
        network_client = aioazure.AzureNetworkClient(subscription_id, resource_group, credentials_file=credentials_file)

        region_monitor = await RegionMonitor.create(region)
        inst_coll_manager = InstanceCollectionManager(db, machine_name_prefix, region_monitor)
        resource_manager = AzureResourceManager(subscription_id, resource_group, ssh_public_key, arm_client, compute_client)

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

        driver = AzureDriver(db,
                             machine_name_prefix,
                             arm_client,
                             compute_client,
                             resources_client,
                             network_client,
                             subscription_id,
                             resource_group,
                             namespace,
                             region_monitor,
                             inst_coll_manager,
                             jpim)

        task_manager.ensure_future(periodically_call(60, driver.delete_orphaned_nics))
        task_manager.ensure_future(periodically_call(60, driver.delete_orphaned_public_ips))

        return driver

    def __init__(self,
                 db: Database,
                 machine_name_prefix: str,
                 arm_client: aioazure.AzureResourceManagerClient,
                 compute_client: aioazure.AzureComputeClient,
                 resources_client: aioazure.AzureResourcesClient,
                 network_client: aioazure.AzureNetworkClient,
                 subscription_id: str,
                 resource_group: str,
                 namespace: str,
                 region_monitor: RegionMonitor,
                 inst_coll_manager: InstanceCollectionManager,
                 job_private_inst_manager: JobPrivateInstanceManager):
        self.db = db
        self.machine_name_prefix = machine_name_prefix
        self.arm_client = arm_client
        self.compute_client = compute_client
        self.resources_client = resources_client
        self.network_client = network_client
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.namespace = namespace
        self.region_monitor = region_monitor
        self.inst_coll_manager = inst_coll_manager
        self.job_private_inst_manager = job_private_inst_manager

    async def shutdown(self) -> None:
        try:
            await self.arm_client.close()
        finally:
            try:
                await self.compute_client.close()
            finally:
                try:
                    await self.resources_client.close()
                finally:
                    await self.network_client.close()

    def _resource_is_orphaned(self, resource_name: str) -> bool:
        instance_name = resource_name.rsplit('-', maxsplit=1)[0]
        return self.inst_coll_manager.get_instance(instance_name) is None

    async def delete_orphaned_nics(self) -> None:
        log.info('deleting orphaned nics')
        async for nic_name in self.resources_client.list_nic_names(self.machine_name_prefix):
            if self._resource_is_orphaned(nic_name):
                try:
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
                    await self.network_client.delete_public_ip(public_ip_name, ignore_not_found=True)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception(f'while deleting orphaned public ip {public_ip_name}')
