import abc
from typing import Dict


class CloudCredentials(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def from_file(credentials_file):
        pass

    @staticmethod
    @abc.abstractmethod
    def default_credentials():
        pass

    @abc.abstractmethod
    async def auth_headers(self) -> Dict[str, str]:
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self):
        raise NotImplementedError
