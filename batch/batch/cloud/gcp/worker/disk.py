import logging
import asyncio
from typing import Dict, Optional

from hailtop.utils import LoggingTimer
from hailtop.aiocloud import aiogoogle

from ....worker.disk import CloudDisk, CloudDiskManager

log = logging.getLogger('disk')


class GCPDisk(CloudDisk):
    def __init__(self, name: str, zone: str, project: str, instance_name: str, size_in_gb: int, mount_path: str):
        assert size_in_gb >= 10
        # disk name must be 63 characters or less
        # https://cloud.google.com/compute/docs/reference/rest/v1/disks#resource:-disk
        # under the information for the name field
        assert len(name) <= 63

        self.compute_client = aiogoogle.GoogleComputeClient(
            project, credentials=aiogoogle.GoogleCredentials.from_file('/worker-key.json')
        )
        self.name = name
        self.zone = zone
        self.project = project
        self.instance_name = instance_name
        self.size_in_gb = size_in_gb
        self.mount_path = mount_path

        self.created = False
        self.attached = False

        self.disk_path = f'/dev/disk/by-id/google-{self.name}'

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

    async def close(self):
        await self.compute_client.close()

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

            await self.compute_client.create_disk(f'/zones/{self.zone}/disks', json=config)
            self.created = True

    async def _attach(self):
        async with LoggingTimer(f'attaching disk {self.name} to {self.instance_name}'):
            config = {
                'source': f'/compute/v1/projects/{self.project}/zones/{self.zone}/disks/{self.name}',
                'autoDelete': True,
                'deviceName': self.name,
            }

            await self.compute_client.attach_disk(
                f'/zones/{self.zone}/instances/{self.instance_name}/attachDisk', json=config
            )
            self.attached = True

    async def _detach(self):
        async with LoggingTimer(f'detaching disk {self.name} from {self.instance_name}'):
            if self.attached:
                await self.compute_client.detach_disk(
                    f'/zones/{self.zone}/instances/{self.instance_name}/detachDisk', params={'deviceName': self.name}
                )
                self.attached = False

    async def _delete(self):
        async with LoggingTimer(f'deleting disk {self.name}'):
            if self.created:
                await self.compute_client.delete_disk(f'/zones/{self.zone}/disks/{self.name}')
                self.created = False

    def __str__(self):
        return self.name


class GCPDiskManager(CloudDiskManager):
    def __init__(self, project: str, zone: str, max_disks: int = 128):
        self.project = project
        self.zone = zone
        self.disk_slots = asyncio.Semaphore(max_disks)

    async def new_disk(self, instance_name: str, disk_name: str, size_in_gb: int, mount_path: str) -> GCPDisk:
        await self.disk_slots.acquire()
        return GCPDisk(disk_name, self.zone, self.project, instance_name, size_in_gb, mount_path)

    async def delete_disk(self, disk: GCPDisk):
        try:
            await disk.delete()
        finally:
            try:
                await disk.close()
            finally:
                self.disk_slots.release()
