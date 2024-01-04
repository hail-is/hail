from contextlib import AsyncExitStack
from types import TracebackType
from typing import Optional, Type, TypeVar, Mapping, Union
import time
import aiohttp
import abc
import logging
from hailtop import httpx
from hailtop.utils import retry_transient_errors, RateLimit, RateLimiter
from .credentials import CloudCredentials, AnonymousCloudCredentials


SessionType = TypeVar('SessionType', bound='BaseSession')
log = logging.getLogger('hailtop.aiocloud.common.session')


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

    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
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
        if hasattr(self, '_session'):
            await self._session.close()
            del self._session


class Session(BaseSession):
    def __init__(
        self,
        *,
        credentials: Union[CloudCredentials, AnonymousCloudCredentials],
        params: Optional[Mapping[str, str]] = None,
        http_session: Optional[httpx.ClientSession] = None,
        **kwargs,
    ):
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
            return await retry_transient_errors(self._request_with_valid_authn, method, url, **kwargs)
        return await self._request_with_valid_authn(method, url, **kwargs)

    async def _request_with_valid_authn(self, method, url, **kwargs):
        while True:
            auth_headers, expiration = await self._credentials.auth_headers_with_expiration()
            if auth_headers:
                if 'headers' in kwargs:
                    kwargs['headers'].update(auth_headers)
                else:
                    kwargs['headers'] = auth_headers
            try:
                return await self._http_session.request(method, url, **kwargs)
            except httpx.ClientResponseError as err:
                if err.status != 401:
                    raise
                if expiration is None or time.time() <= expiration:
                    raise err
                log.info(f'Credentials expired while waiting for request to {url}. We will retry. {err}.')

    async def close(self) -> None:
        async with AsyncExitStack() as stack:
            stack.push_async_callback(self._close_http_session)
            stack.push_async_callback(self._close_credentials)

    async def _close_http_session(self):
        if hasattr(self, '_http_session') and self._owns_http_session:
            await self._http_session.close()
            del self._http_session

    async def _close_credentials(self):
        if hasattr(self, '_credentials'):
            await self._credentials.close()
            del self._credentials
