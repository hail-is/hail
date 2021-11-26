import abc
from typing import Dict


class EmptyCloudCredentials:
    async def close(self):
        pass


class CloudCredentials(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def from_file(credentials_file):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def default_credentials():
        raise NotImplementedError

    @abc.abstractmethod
    async def auth_headers(self) -> Dict[str, str]:
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self):
        raise NotImplementedError
