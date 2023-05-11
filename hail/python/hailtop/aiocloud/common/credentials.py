import abc
from typing import Dict


class Credentials(abc.ABC):
    @abc.abstractmethod
    async def auth_headers(self) -> Dict[str, str]:
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self):
        raise NotImplementedError


class CloudCredentials(Credentials):
    @abc.abstractmethod
    async def access_token(self) -> str:
        raise NotImplementedError


class AnonymousCloudCredentials(CloudCredentials):
    async def auth_headers(self) -> Dict[str, str]:
        return {}

    async def access_token(self) -> str:
        raise ValueError("Cannot request an access token for anonymous credentials")

    async def close(self):
        pass
