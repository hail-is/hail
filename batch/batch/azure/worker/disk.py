import logging

from ...worker.disk import CloudDisk

log = logging.getLogger('disk')


class AzureDisk(CloudDisk):
    async def create(self, labels=None):
        raise NotImplementedError

    async def delete(self):
        raise NotImplementedError

    async def close(self):
        raise NotImplementedError
