from typing import Optional, Callable, Dict

import asyncio
import aiohttp
import logging
import dateutil.parser

from hailtop import aiotools
from hailtop.aiocloud import aioazure
from hailtop.utils import periodically_call
from gear import Database

from ...batch_configuration import AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_REGION
from ...driver.resource_manager import CloudResourceManager, VMDoesNotExist, VMState

from ..resource_utils import worker_properties_to_machine_type
from ..instance_config import AzureInstanceConfig
from .create_instance import create_instance_config
from .resources import delete_orphaned_resources

log = logging.getLogger('resource_manager')


def parse_azure_timestamp(timestamp: Optional[str]) -> Optional[float]:
    if timestamp is None:
        return None
    return dateutil.parser.isoparse(timestamp).timestamp() * 1000


def vm_resources(vm_name) -> Dict[str, str]:
    return {
        'vm_name': vm_name,
        'boot_disk_name': f'{vm_name}-os',
        'data_disk_name': f'{vm_name}-data',
        'nic_name': f'{vm_name}-nic',
        'ip_name': f'{vm_name}-ip',
        'nsg_name': f'{vm_name}-nsg',
    }


class AzureResourceManager(CloudResourceManager):
    def __init__(self, app, machine_name_prefix, make_credentials: Optional[Callable[[], aioazure.AzureCredentials]] = None):
        self.app = app
        self.db: Database = app['db']
        self.machine_name_prefix = machine_name_prefix

        if make_credentials is None:
            def _make_credentials():
                return aioazure.AzureCredentials.from_file('/azure-credentials/credentials.json')
            make_credentials = _make_credentials

        self.arm_client = aioazure.AzureResourcesManagementClient(AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, credentials=make_credentials())
        self.compute_client = aioazure.AzureComputeClient(AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, credentials=make_credentials())
        self.network_client = aioazure.AzureNetworkClient(AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, credentials=make_credentials())
        self.resources_client = aioazure.AzureResourcesClient(AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, credentials=make_credentials())

        self.cloud = 'azure'
        self.default_location = AZURE_REGION

        self.task_manager = aiotools.BackgroundTaskManager()

    async def async_init(self):
        self.task_manager.ensure_future(periodically_call(60, self.delete_orphaned_resources))

    def shutdown(self):
        try:
            self.task_manager.shutdown()
        finally:
            try:
                await self.compute_client.close()
            finally:
                try:
                    await self.arm_client.close()
                finally:
                    try:
                        await self.network_client.close()
                    finally:
                        await self.resources_client.close()

    async def get_live_resources(self, vm_name) -> Dict[str, str]:
        resources = vm_resources(vm_name)

        get_resource_tasks = [
            self.compute_client.get_vm(resources['vm_name']),
            self.compute_client.get_disk(resources['boot_disk_name']),
            self.compute_client.get_disk(resources['data_disk_name']),
            self.network_client.get_nic(resources['nic_name']),
            self.network_client.get_public_ip(resources['ip_name']),
            self.network_client.get_network_security_group(resources['nsg_name']),
        ]

        get_results = await asyncio.gather(*get_resource_tasks, return_exceptions=True)

        def resource_exists(get_result):
            return not (isinstance(get_result, aiohttp.ClientResponseError) and get_result.status == 404)

        existing_resources = {resource_type: resource_name for (resource_type, resource_name), result in zip(list(resources.items()), get_results)
                              if resource_exists(result)}

        return existing_resources

    async def _delete(self, vm_name: str, resources: Optional[Dict[str, str]] = None):
        if resources is None:
            resources = vm_resources(vm_name)

        async def delete_if_exists(f, *args, **kwargs):
            try:
                return await f(*args, **kwargs)
            except aiohttp.ClientResponseError as e:
                if e.status == 404:
                    return
                raise

        try:
            if 'vm_name' in resources:
                await delete_if_exists(self.compute_client.delete_vm_and_wait, resources['vm_name'])

            if 'boot_disk_name' in resources:
                await delete_if_exists(self.compute_client.delete_disk_and_wait, resources['boot_disk_name'])

            if 'data_disk_name' in resources:
                await delete_if_exists(self.compute_client.delete_disk_and_wait, resources['data_disk_name'])

            if 'nic_name' in resources:
                await delete_if_exists(self.network_client.delete_nic_and_wait, resources['nic_name'])

            if 'ip_name' in resources:
                await delete_if_exists(self.network_client.delete_public_ip_and_wait, resources['ip_name'])

            if 'nsg_name' in resources:
                await delete_if_exists(self.network_client.delete_network_security_group_and_wait, resources['nsg_name'])
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(f'error while deleting azure virtual machine {vm_name}', exc_info=True)

    async def delete_vm(self, instance):
        instance_config = instance.instance_config
        assert isinstance(instance_config, AzureInstanceConfig)
        vm_name = instance_config.name

        live_resources = await self.get_live_resources(vm_name)

        if not live_resources:
            raise VMDoesNotExist()

        self.task_manager.ensure_future(self._delete(vm_name, live_resources))

    async def get_vm_state(self, instance) -> VMState:
        instance_config = instance.instance_config
        assert isinstance(instance_config, AzureInstanceConfig)
        vm_name = instance_config.name

        try:
            spec = await self.compute_client.get_vm_instance_view(vm_name)
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                live_resources = await self.get_live_resources(vm_name)
                if not live_resources:
                    raise VMDoesNotExist() from e
                return VMState(VMState.UNKNOWN, None, None)
            raise

        # https://docs.microsoft.com/en-us/azure/virtual-machines/states-billing
        for status in spec['statuses']:
            code = status['code']
            if code == 'ProvisioningState/creating':
                return VMState(VMState.CREATING, spec, instance.time_created)
            if code == 'ProvisioningState/succeeded':
                return VMState(VMState.RUNNING, spec, parse_azure_timestamp(status.get('time')))
            if code in ('ProvisioningState/failed', 'ProvisioningState/deleting', 'ProvisioningState/deleted'):
                return VMState(VMState.TERMINATED, spec, parse_azure_timestamp(status.get('time')))

        log.exception(f'unknown azure codes for {instance}: {spec}')
        return VMState(VMState.UNKNOWN, spec, None)

    def prepare_vm(self,
                   app,
                   machine_name,
                   activation_token,
                   max_idle_time_msecs,
                   local_ssd_data_disk,
                   external_data_disk_size_gb,
                   boot_disk_size_gb,  # pylint: disable=unused-argument
                   preemptible,
                   job_private,
                   machine_type=None,
                   worker_type=None,
                   cores=None,
                   location=None,
                   ) -> Optional[AzureInstanceConfig]:
        assert machine_type or (worker_type and cores)

        if location is None:
            location = AZURE_REGION

        if machine_type is None:
            machine_type = worker_properties_to_machine_type(worker_type, cores, local_ssd_data_disk)

        if location is None:
            location = self.default_location

        return create_instance_config(app, location, machine_name, machine_type, activation_token, max_idle_time_msecs,
                                      local_ssd_data_disk, external_data_disk_size_gb, preemptible, job_private)

    async def create_vm(self, instance_config: AzureInstanceConfig):
        params = {
            'api-version': '2021-04-01'
        }
        await self.arm_client.put(f'/deployments/{instance_config.name}-deployment',
                                  json=instance_config.vm_config,
                                  params=params)
        log.info(f'created machine {instance_config.name}')

    async def delete_orphaned_resources(self):
        await delete_orphaned_resources(self, self.app['inst_coll_manager'])
