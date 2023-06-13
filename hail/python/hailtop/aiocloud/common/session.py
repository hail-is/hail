from types import TracebackType
from typing import Optional, Type, TypeVar, Mapping
import aiohttp
import abc
from hailtop import httpx
from hailtop.utils import retry_transient_errors, RateLimit, RateLimiter
from .credentials import CloudCredentials

SessionType = TypeVar('SessionType', bound='BaseSession')


class BaseSession(abc.ABC):
    @abc.abstractmethod
    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        pass

    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        return await self.request('GET', url, **kwargs)

    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        return await self.request('POST', url, **kwargs)

    async def put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        return await self.request('PUT', url, **kwargs)

    async def patch(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        return await self.request('PATCH', url, **kwargs)

    async def delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        return await self.request('DELETE', url, **kwargs)

    async def head(self, url: str, **kwargs) -> aiohttp.ClientResponse:
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
    _http_session: httpx.ClientSession
    _credentials: CloudCredentials

    def __init__(self,
                 *,
                 credentials: CloudCredentials,
                 params: Optional[Mapping[str, str]] = None,
                 http_session: Optional[httpx.ClientSession] = None,
                 **kwargs):
        if 'raise_for_status' not in kwargs:
            kwargs['raise_for_status'] = True
        self._params = params
        if http_session is not None:
            self._owns_http_session = False
            self._http_session = http_session
        else:
            self._owns_http_session = True
            self._http_session = httpx.ClientSession(**kwargs)
        self._credentials = credentials

    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        auth_headers = await self._credentials.auth_headers()
        if auth_headers:
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
            return await retry_transient_errors(self._http_session.request, method, url, **kwargs)
        return await self._http_session.request(method, url, **kwargs)

    async def close(self) -> None:
        if hasattr(self, '_http_session') and self._owns_http_session:
            await self._http_session.close()
            del self._http_session

        if hasattr(self, '_credentials'):
            await self._credentials.close()
            del self._credentials
