import logging

from ....worker.disk import CloudDisk

log = logging.getLogger('disk')


class AzureDisk(CloudDisk):
    def __init__(self, name: str, instance_name: str, size_in_gb: int, mount_path: str):
        assert size_in_gb >= 10

        self.name = name
        self.instance_name = instance_name
        self.size_in_gb = size_in_gb
        self.mount_path = mount_path

        self._created = False
        self._attached = False

    async def create(self, labels=None):
        raise NotImplementedError

    async def delete(self):
        raise NotImplementedError

    async def close(self):
        raise NotImplementedError
