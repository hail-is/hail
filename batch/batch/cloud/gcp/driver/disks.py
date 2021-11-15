import logging
import aiohttp
import asyncio
from typing import List

from hailtop.aiocloud import aiogoogle
from hailtop.utils import time_msecs, parse_timestamp_msecs

from ....driver.instance_collection import InstanceCollectionManager

log = logging.getLogger('disks')


async def delete_orphaned_disks(compute_client: aiogoogle.GoogleComputeClient, zones: List[str],
                                inst_coll_manager: InstanceCollectionManager, namespace: str):
    log.info('deleting orphaned disks')

    params = {'filter': f'(labels.namespace = {namespace})'}

    for zone in zones:
        async for disk in await compute_client.list(f'/zones/{zone}/disks', params=params):
            disk_name = disk['name']
            instance_name = disk['labels']['instance-name']
            instance = inst_coll_manager.get_instance(instance_name)

            creation_timestamp_msecs = parse_timestamp_msecs(disk.get('creationTimestamp'))
            last_attach_timestamp_msecs = parse_timestamp_msecs(disk.get('lastAttachTimestamp'))
            last_detach_timestamp_msecs = parse_timestamp_msecs(disk.get('lastDetachTimestamp'))

            now_msecs = time_msecs()
            if instance is None:
                log.exception(f'deleting disk {disk_name} from instance that no longer exists')
            elif (last_attach_timestamp_msecs is None
                  and now_msecs - creation_timestamp_msecs > 60 * 60 * 1000):
                log.exception(f'deleting disk {disk_name} that has not attached within 60 minutes')
            elif (last_detach_timestamp_msecs is not None
                  and now_msecs - last_detach_timestamp_msecs > 5 * 60 * 1000):
                log.exception(f'deleting detached disk {disk_name} that has not been cleaned up within 5 minutes')
            else:
                continue

            try:
                await compute_client.delete_disk(f'/zones/{zone}/disks/{disk_name}')
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if isinstance(e, aiohttp.ClientResponseError) and e.status == 404:  # pylint: disable=no-member
                    continue
                log.exception(f'error while deleting orphaned disk {disk_name}')
