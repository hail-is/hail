import abc
from typing import Dict


class CloudCredentials(abc.ABC):
    @abc.abstractmethod
    async def auth_headers(self) -> Dict[str, str]:
        raise NotImplementedError

    @abc.abstractmethod
    async def access_token(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self):
        raise NotImplementedError


class AnonymousCloudCredentials:
    async def auth_headers(self) -> Dict[str, str]:
        return {}

    async def close(self):
        pass
