from typing import Dict, List, Optional

import logging
import asyncio

import aiohttp
from hailtop.aiocloud import aioazure
from hailtop.utils import sleep_and_backoff, LoggingTimer

from ....worker.disk import CloudDisk, CloudDiskManager
from ...azure.instance_config import AzureSlimInstanceConfig

log = logging.getLogger('disk')

KNOWN_DISK_STATUS_CODES = [
    'ProvisioningState/creating',
    'ProvisioningState/succeeded',
    'ProvisioningState/failed',
    'ProvisioningState/deleting',
    'ProvisioningState/deleted',
]


class AzureDisk(CloudDisk):
    def __init__(self, disk_manager: 'AzureDiskManager', name: str, instance_name: str, size_in_gb: int, mount_path: str, lun: int):
        # FIXME: how to handle the rounding up to the nearest power of two!
        assert size_in_gb >= 10
        # https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/resource-name-rules#microsoftcompute
        assert 1 <= len(name) <= 80, name

        self.disk_manager = disk_manager
        self.name = name
        self.instance_name = instance_name
        self.size_in_gb = size_in_gb
        self.mount_path = mount_path
        self.device_path = f'/dev/disk/azure/scsi1/lun{lun}'
        self.lun = lun

        self.compute_client = aioazure.AzureComputeClient(disk_manager.subscription_id, disk_manager.resource_group)

        self.created = False
        self.attached = False

    def spec(self):
        return {
            'createOption': 'Empty',
            'deleteOption': 'Delete',
            'diskSizeGB': self.size_in_gb,
            'lun': self.lun,
            'managedDisk': {
                'storageAccountType': 'Premium_LRS'
            },
            'name': self.name
        }

    def _extract_disk_from_disks(self, disks: List[dict]) -> Optional[dict]:
        matches = [disk['name'] == self.name for disk in disks]
        if len(matches) == 0:
            return None
        return matches[0]

    async def _create_and_attach(self):
        async with LoggingTimer(f'creating disk {self.name}'):
            async with self.disk_manager.update_lock:
                new_vm_disk_config = self.disk_manager._add_disk(self)
                await self.compute_client.patch(f'/virtualMachines/{self.instance_name}', json=new_vm_disk_config)

            delay = 1
            while True:
                spec = await self.compute_client.get(f'/virtualMachines/{self.instance_name}/instanceView')
                disk = self._extract_disk_from_disks(spec['disks'])

                if disk is not None:
                    statuses = disk['statuses']
                    for status in statuses:
                        if status['code'] == 'ProvisioningState/creating':
                            self.created = True

                        if status['code'] == 'ProvisioningState/succeeded':
                            self.created = True
                            self.attached = True
                            return

                        if status['code'] in ('ProvisioningState/failed', 'ProvisioningState/deleting', 'ProvisioningState/deleted'):
                            raise Exception(f'disk creation for instance {self.instance_name} failed with code {status["code"]}: {disk}')

                        if status['code'] not in KNOWN_DISK_STATUS_CODES:
                            raise Exception(f'unknown disk status code: {disk}')

                delay = await sleep_and_backoff(delay)

    async def _detach(self):
        if self.attached:
            async with LoggingTimer(f'detaching disk {self.name} from {self.instance_name}'):
                async with self.disk_manager.update_lock:
                    # https://docs.microsoft.com/en-us/rest/api/compute/virtual-machines/update#update-a-vm-by-detaching-data-disk
                    new_vm_disk_config = self.disk_manager._remove_disk(self)
                    await self.compute_client.patch(f'/virtualMachines/{self.instance_name}', json=new_vm_disk_config)

                delay = 1
                while True:
                    spec = await self.compute_client.get(f'/virtualMachines/{self.instance_name}/instanceView')
                    disk_status = self._extract_disk_from_disks(spec['disks'])
                    if disk_status is None:
                        self.attached = False
                        return
                    delay = await sleep_and_backoff(delay, max_delay=10)

    async def _delete(self):
        async with LoggingTimer(f'deleting disk {self.name}'):
            if self.created:
                try:
                    params = {'api-version': '2020-12-01'}
                    await self.compute_client.delete(f'/disks/{self.name}', params=params)
                except aiohttp.ClientResponseError as e:
                    if e.status in (204, 404):  # https://docs.microsoft.com/en-us/rest/api/compute/disks/delete#response
                        pass

                self.created = False

    async def create(self, labels: Optional[Dict[str, str]] = None):
        del labels  # disks automatically inherit the VM's labels
        await self._create_and_attach()
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


class AzureDiskManager(CloudDiskManager):
    def __init__(self, machine_name: str, instance_config: AzureSlimInstanceConfig, subscription_id: str, resource_group: str):
        self.subscription_id = subscription_id
        self.resource_group = resource_group

        self.disk_mapping: Dict[str, dict] = {config['name']: config for config in instance_config.data_disks(machine_name)}

        self.update_lock = asyncio.Lock()
        self.lun_queue: asyncio.Queue[int] = asyncio.Queue()

        start_lun = 0 if instance_config.local_ssd_data_disk else 1  # FIXME: double check this works!
        max_lun = min(32, instance_config.cores * 2)  # https://docs.microsoft.com/en-us/azure/virtual-machines/ddv4-ddsv4-series
        for lun in range(start_lun, max_lun):
            self.lun_queue.put_nowait(lun)

    def vm_disk_config(self):
        return {
            'properties': {
                'storageProfile': {
                    'dataDisks': list(self.disk_mapping.values())
                }
            }
        }

    def _add_disk(self, disk: AzureDisk) -> dict:
        self.disk_mapping[disk.name] = disk.spec()
        return self.vm_disk_config()

    def _remove_disk(self, disk: AzureDisk) -> dict:
        self.disk_mapping.pop(disk.name)
        return self.vm_disk_config()

    async def new_disk(self, instance_name: str, disk_name: str, size_in_gb: int, mount_path: str) -> AzureDisk:
        lun = await self.lun_queue.get()
        return AzureDisk(self, disk_name, instance_name, size_in_gb, mount_path, lun)

    async def delete_disk(self, disk: AzureDisk):  # type: ignore[override]
        try:
            await disk.delete()
        finally:
            try:
                if not disk.attached:
                    self.lun_queue.put_nowait(disk.lun)
                else:
                    log.exception(f'disk {disk.name} did not detach cleanly. not releasing lun {disk.lun}')
            finally:
                await disk.close()
