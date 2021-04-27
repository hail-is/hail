from types import TracebackType
from typing import Optional, Type, TypeVar, Mapping
import abc
import aiohttp
from hailtop.utils import request_retry_transient_errors, RateLimit, RateLimiter
from .credentials import Credentials
from .access_token import AccessToken

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
    _session: BaseSession

    def __init__(self, *, session: BaseSession, rate_limit: RateLimit):
        self._session = session
        self._rate_limiter = RateLimiter(rate_limit)

    async def request(self, method: str, url: str, **kwargs):
        async with self._rate_limiter:
            return await self._session.request(method, url, **kwargs)

    async def close(self) -> None:
        if hasattr(self._session, '_session'):
            await self._session.close()
            del self._session


class Session(BaseSession):
    _session: aiohttp.ClientSession
    _access_token: AccessToken

    def __init__(self, *, credentials: Credentials = None, params: Optional[Mapping[str, str]] = None, **kwargs):
        if credentials is None:
            credentials = Credentials.default_credentials()

        if 'raise_for_status' not in kwargs:
            kwargs['raise_for_status'] = True
        self._params = params
        self._session = aiohttp.ClientSession(**kwargs)
        self._access_token = AccessToken(credentials)

    async def request(self, method: str, url: str, **kwargs):
        auth_headers = await self._access_token.auth_headers(self._session)
        if 'headers' in kwargs:
            kwargs['headers'].update(auth_headers)
        else:
            kwargs['headers'] = auth_headers

        if self._params:
            if 'params' in kwargs:
                request_params = kwargs['params']
            else:
                request_params = {}
                kwargs['params'] = request_params
            for k, v in self._params.items():
                if k not in request_params:
                    request_params[k] = v

        # retry by default
        retry = kwargs.pop('retry', True)
        if retry:
            return await request_retry_transient_errors(self._session, method, url, **kwargs)
        return await self._session.request(method, url, **kwargs)

    async def close(self) -> None:
        if hasattr(self._session, '_session'):
            await self._session.close()
            del self._session
        del self._access_token
