import logging
import aiohttp

from hailtop.utils import time_msecs, parse_timestamp_msecs

from ...batch_configuration import DEFAULT_NAMESPACE
from ...driver.disk_monitor import BaseDiskMonitor

log = logging.getLogger('disk_monitor')


class DiskMonitor(BaseDiskMonitor):
    async def delete_orphaned_disks(self):
        log.info('deleting orphaned disks')

        params = {'filter': f'(labels.namespace = {DEFAULT_NAMESPACE})'}

        for zone in self.compute_manager.zone_monitor.zones:
            async for disk in await self.compute_manager.compute_client.list(f'/zones/{zone}/disks', params=params):
                disk_name = disk['name']
                instance_name = disk['labels']['instance-name']
                instance = self.app['inst_coll_manager'].get_instance(instance_name)

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
                    await self.compute_manager.compute_client.delete_disk(f'/zones/{zone}/disks/{disk_name}')
                except aiohttp.ClientResponseError as e:
                    if e.status == 404:
                        continue
                    log.exception(f'error while deleting orphaned disk {disk_name}')
