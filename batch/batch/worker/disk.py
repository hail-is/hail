import logging

from hailtop.utils import check_shell_output, LoggingTimer

log = logging.getLogger('disk')


class Disk:
    def __init__(self, compute_client, name, zone, project, instance_name, size_in_gb, mount_path):
        assert size_in_gb >= 10
        # disk name must be 63 characters or less
        # https://cloud.google.com/compute/docs/reference/rest/v1/disks#resource:-disk
        # under the information for the name field
        assert len(name) <= 63

        self.compute_client = compute_client
        self.name = name
        self.zone = zone
        self.project = project
        self.instance_name = instance_name
        self.size_in_gb = size_in_gb
        self.mount_path = mount_path

        self.disk_path = f'/dev/disk/by-id/google-{self.name}'

    async def __aenter__(self, labels=None):
        await self.create(labels)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.delete()

    async def create(self, labels=None):
        await self._create(labels)
        await self._attach()
        await self._format()

    async def delete(self):
        try:
            await self._detach()
        finally:
            await self._delete()

    async def _format(self):
        await check_shell_output(f'mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard {self.disk_path}')
        await check_shell_output(f'mkdir -p {self.mount_path}')
        await check_shell_output(f'mount -o discard,defaults {self.disk_path} {self.mount_path}')
        await check_shell_output(f'chmod a+w {self.mount_path}')

    async def _create(self, labels=None):
        async with LoggingTimer(f'creating disk {self.name}'):
            if labels is None:
                labels = {}

            config = {
                'name': self.name,
                'sizeGb': f'{self.size_in_gb}',
                'type': f'zones/{self.zone}/diskTypes/pd-ssd',
                'labels': labels
            }

            await self.compute_client.create_disk(f'/zones/{self.zone}/disks',
                                                  json=config)

    async def _attach(self):
        async with LoggingTimer(f'attaching disk {self.name} to {self.instance_name}'):
            config = {
                'source': f'/compute/v1/projects/{self.project}/zones/{self.zone}/disks/{self.name}',
                'autoDelete': True,
                'deviceName': self.name
            }

            await self.compute_client.attach_disk(f'/zones/{self.zone}/instances/{self.instance_name}/attachDisk',
                                                  json=config)

    async def _detach(self):
        async with LoggingTimer(f'detaching disk {self.name} from {self.instance_name}'):
            await self.compute_client.detach_disk(f'/zones/{self.zone}/instances/{self.instance_name}/detachDisk',
                                                  params={'deviceName': self.name})

    async def _delete(self):
        async with LoggingTimer(f'deleting disk {self.name}'):
            await self.compute_client.delete_disk(f'/zones/{self.zone}/disks/{self.name}')

    def __str__(self):
        return self.name
