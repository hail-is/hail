import abc
from typing import Dict, Optional, Tuple


class CloudCredentials(abc.ABC):
    @abc.abstractmethod
    async def auth_headers_with_expiration(self) -> Tuple[Dict[str, str], Optional[float]]:
        """Return HTTP authentication headers and the time of expiration in seconds since the epoch (Unix time).

        None indicates a non-expiring credentials."""
        raise NotImplementedError

    @abc.abstractmethod
    async def access_token_with_expiration(self) -> Tuple[str, Optional[float]]:
        """Return an access token and the time of expiration in seconds since the epoch (Unix time).

        None indicates a non-expiring credentials."""
        raise NotImplementedError

    async def auth_headers(self) -> Dict[str, str]:
        headers, _ = await self.auth_headers_with_expiration()
        return headers

    async def access_token(self) -> str:
        access_token, _ = await self.access_token_with_expiration()
        return access_token

    @abc.abstractmethod
    async def close(self):
        raise NotImplementedError


class AnonymousCloudCredentials:
    async def auth_headers_with_expiration(self) -> Tuple[Dict[str, str], Optional[float]]:
        return {}, None

    async def auth_headers(self) -> Dict[str, str]:
        return {}

    async def close(self):
        pass
