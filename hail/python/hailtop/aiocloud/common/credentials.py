import abc
from typing import Dict


class CloudCredentials(abc.ABC):
    @staticmethod
    def from_file(credentials_file):
        raise NotImplementedError

    @staticmethod
    def default_credentials():
        raise NotImplementedError

    @abc.abstractmethod
    async def auth_headers(self) -> Dict[str, str]:
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self):
        raise NotImplementedError


class AnonymousCloudCredentials(CloudCredentials):
    async def auth_headers(self) -> Dict[str, str]:
        return {}

    async def close(self):
        pass
