import abc
import logging

log = logging.getLogger('disk')


class CloudDisk(abc.ABC):
    name: str

    async def __aenter__(self, labels=None):
        await self.create(labels)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.delete()
        await self.close()

    @abc.abstractmethod
    async def create(self, labels=None):
        pass

    @abc.abstractmethod
    async def delete(self):
        pass

    @abc.abstractmethod
    async def close(self):
        pass
