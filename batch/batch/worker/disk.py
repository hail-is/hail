import uuid
import logging

from hailtop.utils import sleep_and_backoff, check_shell

log = logging.getLogger('disk')


class Disk:
    def __init__(self, compute_client, name, zone, project, instance_name, size_in_gb, mount_path):
        assert size_in_gb >= 10

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
        await self._detach()
        await self._delete()

    async def _format(self):
        code = f'''
mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard {self.disk_path}
mkdir -p {self.mount_path}
mount -o discard,defaults {self.disk_path} {self.mount_path}
chmod a+w {self.mount_path}
'''
        await check_shell(code, echo=True)

    async def _create(self, labels=None):
        log.info(f'creating disk {self.name}')
        if labels is None:
            labels = {}

        config = {
            'name': self.name,
            'sizeGb': f'{self.size_in_gb}',
            'type': f'zones/{self.zone}/diskTypes/pd-ssd',
            'labels': labels
        }

        resp = None
        delay = 0.2
        create_uuid = str(uuid.uuid4())
        while resp is None or resp['status'] != 'DONE':  # pylint: disable=unsubscriptable-object
            resp = await self.compute_client.post(f'/zones/{self.zone}/disks',
                                                  json=config,
                                                  params={'requestId': create_uuid})
            log.info(resp)
            delay = await sleep_and_backoff(delay)
        log.info(f'created disk {self.name}')

    async def _attach(self):
        log.info(f'attaching disk {self.name} to {self.instance_name}')
        config = {
            'source': f'/compute/v1/projects/{self.project}/zones/{self.zone}/disks/{self.name}',
            'autoDelete': True,
            'deviceName': self.name
        }

        resp = None
        delay = 0.2
        attach_uuid = str(uuid.uuid4())
        while resp is None or resp['status'] != 'DONE':  # pylint: disable=unsubscriptable-object
            resp = await self.compute_client.post(
                f'/zones/{self.zone}/instances/{self.instance_name}/attachDisk',
                json=config,
                params={'requestId': attach_uuid})
            delay = await sleep_and_backoff(delay)
        log.info(f'attached disk {self.name} to {self.instance_name}')

    async def _detach(self):
        log.info(f'detaching disk {self.name} from {self.instance_name}')
        resp = None
        delay = 0.2
        detach_uuid = str(uuid.uuid4())
        while resp is None or resp['status'] != 'DONE':  # pylint: disable=unsubscriptable-object
            resp = await self.compute_client.post(
                f'/zones/{self.zone}/instances/{self.instance_name}/detachDisk',
                params={'requestId': detach_uuid,
                        'deviceName': self.name})
            delay = await sleep_and_backoff(delay)
        log.info(f'detached disk {self.name} from {self.instance_name}')

    async def _delete(self):
        log.info(f'deleting disk {self.name}')
        resp = None
        delay = 0.2
        delete_uuid = str(uuid.uuid4())
        while resp is None or resp['status'] != 'DONE':  # pylint: disable=unsubscriptable-object
            resp = await self.compute_client.delete(
                f'/zones/{self.zone}/disks/{self.name}',
                params={'requestId': delete_uuid})
            delay = await sleep_and_backoff(delay)
        log.info(f'deleted disk {self.name}')

    def __str__(self):
        return self.name
