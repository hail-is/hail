import logging
import os
from typing import Dict, Optional

from hailtop.aiocloud import aiogoogle
from hailtop.utils import CalledProcessError, LoggingTimer, check_shell_output, retry_all_errors_n_times

from ....worker.disk import CloudDisk

log = logging.getLogger('disk')


class GCPDisk(CloudDisk):
    def __init__(
        self,
        name: str,
        zone: str,
        project: str,
        instance_name: str,
        size_in_gb: int,
        mount_path: str,
        compute_client: aiogoogle.GoogleComputeClient,  # BORROWED
    ):
        assert size_in_gb >= 10
        # disk name must be 63 characters or less
        # https://cloud.google.com/compute/docs/reference/rest/v1/disks#resource:-disk
        # under the information for the name field
        assert len(name) <= 63

        self.name = name
        self.zone = zone
        self.project = project
        self.instance_name = instance_name
        self.size_in_gb = size_in_gb
        self.mount_path = mount_path
        self.compute_client = compute_client

        self._created = False
        self._attached = False

        self.disk_path = f'/dev/disk/by-id/google-{self.name}'

        self.last_response = None

    async def create(self, labels: Optional[Dict[str, str]] = None):
        await self._create(labels)
        await self._attach()
        await self._format()

    async def delete(self):
        try:
            await self._unmount()
        finally:
            try:
                await self._detach()
            finally:
                await self._delete()

    async def _unmount(self):
        if self._attached:
            await retry_all_errors_n_times(
                max_errors=10, msg=f'error while unmounting disk {self.name}', error_logging_interval=3
            )(check_shell_output, f'umount -v {self.disk_path} {self.mount_path}')

    async def _format(self):
        async def format_disk():
            try:
                await check_shell_output(
                    f'mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard {self.disk_path}'
                )
                await check_shell_output(f'mkdir -p {self.mount_path}')
                await check_shell_output(f'mount -o discard,defaults {self.disk_path} {self.mount_path}')
                await check_shell_output(f'chmod a+w {self.mount_path}')
            except CalledProcessError:
                try:
                    outerr = await check_shell_output(f'ls {os.path.dirname(self.disk_path)}')
                    log.info(f'debugging info while formatting disk {self.name}: {outerr}\n{self.last_response}')
                except CalledProcessError:
                    log.exception(
                        f'error while getting limited debugging info for formatting disk {self.name}:\n{self.last_response}'
                    )

                raise

        await retry_all_errors_n_times(
            max_errors=10, msg=f'error while formatting disk {self.name}', error_logging_interval=3
        )(format_disk)

    async def _create(self, labels: Optional[Dict[str, str]] = None):
        async with LoggingTimer(f'creating disk {self.name}'):
            if labels is None:
                labels = {}

            config = {
                'name': self.name,
                'sizeGb': f'{self.size_in_gb}',
                'type': f'zones/{self.zone}/diskTypes/pd-ssd',
                'labels': labels,
            }

            self.last_response = await self.compute_client.create_disk(f'/zones/{self.zone}/disks', json=config)
            self._created = True

    async def _attach(self):
        async with LoggingTimer(f'attaching disk {self.name} to {self.instance_name}'):
            config = {
                'source': f'/compute/v1/projects/{self.project}/zones/{self.zone}/disks/{self.name}',
                'autoDelete': True,
                'deviceName': self.name,
            }

            try:
                self.last_response = await self.compute_client.attach_disk(
                    f'/zones/{self.zone}/instances/{self.instance_name}/attachDisk', json=config
                )
            except aiogoogle.client.compute_client.GCPOperationError as e:
                if e.status == 400:
                    assert e.error_messages and e.error_codes
                    if all(self.instance_name in em for em in e.error_messages) and all(
                        em == 'RESOURCE_IN_USE_BY_ANOTHER_RESOURCE' for em in e.error_codes
                    ):
                        pass
                    else:
                        raise

            self._attached = True

    async def _detach(self):
        async with LoggingTimer(f'detaching disk {self.name} from {self.instance_name}'):
            if self._attached:
                self.last_response = await self.compute_client.detach_disk(
                    f'/zones/{self.zone}/instances/{self.instance_name}/detachDisk', params={'deviceName': self.name}
                )

    async def _delete(self):
        async with LoggingTimer(f'deleting disk {self.name}'):
            if self._created:
                self.last_response = await self.compute_client.delete_disk(f'/zones/{self.zone}/disks/{self.name}')

    def __str__(self):
        return self.name
