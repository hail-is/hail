from typing import Dict, Optional

import abc
import logging

from hailtop.utils import check_shell_output, retry_all_errors_n_times

log = logging.getLogger('disk')


class CloudDisk(abc.ABC):
    name: str
    mount_path: str
    device_path: str
    disk_path: str
    attached: bool
    created: bool

    async def __aenter__(self, labels=None):
        await self.create(labels)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.delete()
        await self.close()

    @abc.abstractmethod
    async def create(self, labels: Optional[Dict[str, str]] = None):
        pass

    @abc.abstractmethod
    async def delete(self):
        pass

    @abc.abstractmethod
    async def close(self):
        pass

    async def _format(self):
        async def format_disk():
            await check_shell_output(
                f'mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard {self.disk_path}'
            )
            await check_shell_output(f'mkdir -p {self.mount_path}')
            await check_shell_output(f'mount -o discard,defaults {self.disk_path} {self.mount_path}')
            await check_shell_output(f'chmod a+w {self.mount_path}')

        await retry_all_errors_n_times(
            max_errors=10, msg=f'error while formatting disk {self.name}', error_logging_interval=3
        )(format_disk)

    async def _unmount(self):
        if self.attached:
            await retry_all_errors_n_times(
                max_errors=10, msg=f'error while unmounting disk {self.name}', error_logging_interval=3
            )(check_shell_output, f'umount -v {self.disk_path} {self.mount_path}')


class CloudDiskManager(abc.ABC):
    @abc.abstractmethod
    async def new_disk(self, instance_name: str, disk_name: str, size_in_gb: int, mount_path: str) -> CloudDisk:
        pass

    @abc.abstractmethod
    async def delete_disk(self, disk: CloudDisk):
        pass
