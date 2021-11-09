import asyncio
import logging

import aiohttp
from hailtop.aiocloud import aioazure

from ....driver.instance_collection import InstanceCollectionManager

log = logging.getLogger('networks')


async def delete_orphaned_nics(
        resources_client: aioazure.AzureResourcesClient,
        network_client: aioazure.AzureNetworkClient,
        inst_coll_manager: InstanceCollectionManager,
        machine_name_prefix: str):
    log.info('deleting orphaned nics')

    resource_filter = f"resourceType eq 'Microsoft.Network/networkInterfaces' and substringof('{machine_name_prefix}',name)"
    async for resource in await resources_client.list_resources(filter=resource_filter):
        resource_name = resource['name']
        instance_name = resource_name.rsplit('-', maxsplit=1)[0]
        instance = inst_coll_manager.get_instance(instance_name)
        if instance is None:
            try:
                await network_client.delete(f'/networkInterfaces/{resource_name}')
                log.info(f'deleted orphaned nic {resource_name}')
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if isinstance(e, aiohttp.ClientResponseError) and e.status == 404:
                    continue
                log.exception(f'error while deleting {resource_name}')


async def delete_orphaned_public_ips(
        resources_client: aioazure.AzureResourcesClient,
        network_client: aioazure.AzureNetworkClient,
        inst_coll_manager: InstanceCollectionManager,
        machine_name_prefix: str):
    log.info('deleting orphaned public ip addresses')

    resource_filter = f"resourceType eq 'Microsoft.Network/publicIPAddresses' and substringof('{machine_name_prefix}',name)"
    async for resource in await resources_client.list_resources(filter=resource_filter):
        resource_name = resource['name']
        instance_name = resource_name.rsplit('-', maxsplit=1)[0]
        instance = inst_coll_manager.get_instance(instance_name)
        if instance is None:
            try:
                await network_client.delete(f'/networkInterfaces/{resource_name}')
                log.info(f'deleted orphaned public ip {resource_name}')
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if isinstance(e, aiohttp.ClientResponseError) and e.status == 404:
                    continue
                log.exception(f'error while deleting {resource_name}')
