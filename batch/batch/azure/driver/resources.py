import asyncio
import logging
from typing import TYPE_CHECKING

from ...batch_configuration import DEFAULT_NAMESPACE
from ...driver.resource_manager import VMDoesNotExist

log = logging.getLogger('resources')

if TYPE_CHECKING:
    from .resource_manager import AzureResourceManager  # pylint: disable=cyclic-import


async def delete_orphaned_resources(resource_manager: 'AzureResourceManager', inst_coll_manager):
    log.info('deleting orphaned resources')

    # Check for orphaned network security groups as that's the last dependency to delete
    # Cannot filter on both tag names
    filter = f"tagName eq 'namespace' and tagValue eq '{DEFAULT_NAMESPACE}'"
    filter = "resourceType eq 'Microsoft.Network/networkSecurityGroups'"
    resources = await resource_manager.resources_client.list_resources(filter=filter)
    async for resource in resources:
        namespace = resource['tags'].get('namespace')
        if namespace and namespace == DEFAULT_NAMESPACE:
            vm_name = resource['name'].rsplit('-')[0]

            instance = inst_coll_manager.get_instance(vm_name)
            if instance is None:
                continue

            try:
                await resource_manager.delete_vm(instance)
            except asyncio.CancelledError:
                raise
            except VMDoesNotExist:
                pass
            except Exception:
                log.exception(f'error while deleting orphaned vm {vm_name}')
