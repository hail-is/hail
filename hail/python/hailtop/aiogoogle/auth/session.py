from types import TracebackType
from typing import Optional, Type, TypeVar
import abc
import aiohttp
from hailtop.utils import RateLimit, RateLimiter
from .credentials import Credentials
from .access_token import AccessToken

from ... import httpx

SessionType = TypeVar('SessionType', bound='BaseSession')


class BaseSession(abc.ABC):
    @abc.abstractmethod
    async def request(self, method: str, url: str, **kwargs):
        pass

    async def get(self, url: str, **kwargs):
        return await self.request('GET', url, **kwargs)

    async def post(self, url: str, **kwargs):
        return await self.request('POST', url, **kwargs)

    async def put(self, url: str, **kwargs):
        return await self.request('PUT', url, **kwargs)

    async def delete(self, url: str, **kwargs):
        return await self.request('DELETE', url, **kwargs)

    async def head(self, url: str, **kwargs):
        return await self.request('HEAD', url, **kwargs)

    async def close(self) -> None:
        pass

    async def __aenter__(self: SessionType) -> SessionType:
        return self

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        await self.close()


class RateLimitedSession(BaseSession):
    _session: Optional[BaseSession]

    def __init__(self, *, session: BaseSession, rate_limit: RateLimit):
        self._session = session
        self._rate_limiter = RateLimiter(rate_limit)

    async def request(self, method: str, url: str, **kwargs):
        async with self._rate_limiter:
            return await self._session.request(method, url, **kwargs)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


class Session(BaseSession):
    _session: Optional[httpx.ClientSession]
    _access_token: Optional[AccessToken]

    def __init__(self, *, credentials: Credentials = None, **kwargs):
        if credentials is None:
            credentials = Credentials.default_credentials()

        self._session = httpx.client_session(**kwargs)
        self._access_token = AccessToken(credentials)

    async def request(self, method: str, url: str, **kwargs):
        assert self._access_token is not None
        assert self._session is not None
        auth_headers = await self._access_token.auth_headers(self._session)
        if 'headers' in kwargs:
            kwargs['headers'].update(auth_headers)
        else:
            kwargs['headers'] = auth_headers
        return self._session.request(method, url, **kwargs)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._access_token = None
