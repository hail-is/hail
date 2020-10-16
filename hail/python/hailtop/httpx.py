from typing import Dict, Any, List, Tuple, Optional, Type, Union
from types import TracebackType
import aiohttp
import socket
import logging
from hailtop.config import get_deploy_config
from .tls import internal_client_ssl_context, external_client_ssl_context
from .utils import async_to_blocking, retry_transient_errors

log = logging.getLogger('hailtop.httpx')

HailAsyncClientSession = Union[aiohttp.ClientSession, 'RetryingClientSession']


def client_session(*args,
                   retry_transient=True,
                   raise_for_status=True,
                   **kwargs) -> HailAsyncClientSession:
    assert 'connector' not in kwargs
    if get_deploy_config().location() == 'external':
        kwargs['connector'] = aiohttp.TCPConnector(
            ssl=external_client_ssl_context())
    else:
        kwargs['connector'] = aiohttp.TCPConnector(
            ssl=internal_client_ssl_context(),
            resolver=HailResolver())
    if 'timeout' not in kwargs:
        kwargs['timeout'] = aiohttp.ClientTimeout(total=5)
    kwargs['raise_for_status'] = raise_for_status
    session = aiohttp.ClientSession(*args, **kwargs)
    if retry_transient:
        return RetryingClientSession(session)
    return session


class ResponseManager:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc: Optional[BaseException],
                        tb: Optional[TracebackType]) -> None:
        self.response.close()


class RetryingClientSession:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def request(self,
                      method: str,
                      url: aiohttp.typedefs.StrOrURL,
                      **kwargs: Any) -> ResponseManager:
        return ResponseManager(await retry_transient_errors(
            self.session._request, method, url, **kwargs))

    async def get(self,
                  url: aiohttp.typedefs.StrOrURL,
                  *,
                  allow_redirects: bool = True,
                  **kwargs: Any) -> ResponseManager:
        return await self.request(
            aiohttp.hdrs.METH_GET, url, allow_redirects=allow_redirects, **kwargs)

    async def options(self,
                      url: aiohttp.typedefs.StrOrURL,
                      *,
                      allow_redirects: bool = True,
                      **kwargs: Any) -> ResponseManager:
        return await self.request(
            aiohttp.hdrs.METH_OPTIONS, url, allow_redirects=allow_redirects, **kwargs)

    async def head(self,
                   url: aiohttp.typedefs.StrOrURL,
                   *,
                   allow_redirects: bool = False,
                   **kwargs: Any) -> ResponseManager:
        return await self.request(
            aiohttp.hdrs.METH_HEAD, url, allow_redirects=allow_redirects, **kwargs)

    async def post(self,
                   url: aiohttp.typedefs.StrOrURL,
                   *,
                   data: Any = None, **kwargs: Any) -> ResponseManager:
        return await self.request(
            aiohttp.hdrs.METH_POST, url, data=data, **kwargs)

    async def put(self,
                  url: aiohttp.typedefs.StrOrURL,
                  *,
                  data: Any = None,
                  **kwargs: Any) -> ResponseManager:
        return await self.request(
            aiohttp.hdrs.METH_PUT, url, data=data, **kwargs)

    async def patch(self,
                    url: aiohttp.typedefs.StrOrURL,
                    *,
                    data: Any = None,
                    **kwargs: Any) -> ResponseManager:
        return await self.request(
            aiohttp.hdrs.METH_PATCH, url, data=data, **kwargs)

    async def delete(self,
                     url: aiohttp.typedefs.StrOrURL,
                     **kwargs: Any) -> ResponseManager:
        return await self.request(
            aiohttp.hdrs.METH_DELETE, url, **kwargs)

    async def close(self) -> None:
        await self.session.close()

    @property
    def closed(self) -> bool:
        return self.session.closed

    @property
    def cookie_jar(self) -> aiohttp.abc.AbstractCookieJar:
        return self.session.cookie_jar

    @property
    def version(self) -> Tuple[int, int]:
        return self.session.version

    async def __aenter__(self) -> 'RetryingClientSession':
        self.session = await self.session.__aenter__()
        return self

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        await self.session.close()


class HailResolver(aiohttp.abc.AbstractResolver):
    """Use Hail in-cluster DNS with fallback."""
    def __init__(self):
        self.dns = aiohttp.AsyncResolver()
        self.deploy_config = get_deploy_config()

    async def resolve(self, host: str, port: int, family: int) -> List[Dict[str, Any]]:
        if family in (0, socket.AF_INET, socket.AF_INET6):
            maybe_address_and_port = await self.deploy_config.maybe_address(host)
            if maybe_address_and_port is not None:
                address, resolved_port = maybe_address_and_port
                return [{'hostname': host,
                         'host': address,
                         'port': resolved_port,
                         'family': family,
                         'proto': 0,
                         'flags': 0}]
        return await self.dns.resolve(host, port, family)

    async def close(self) -> None:
        await self.dns.close()


class BlockingClientResponse:
    def __init__(self, client_response):
        self.client_response = client_response

    def read(self) -> bytes:
        return async_to_blocking(self.client_response.read())

    def text(self, encoding: Optional[str] = None, errors: str = 'strict') -> str:
        return async_to_blocking(self.client_response.text(
            encoding=encoding, errors=errors))

    def json(self, *,
             encoding: str = None,
             loads: aiohttp.typedefs.JSONDecoder = aiohttp.typedefs.DEFAULT_JSON_DECODER,
             content_type: Optional[str] = 'application/json') -> Any:
        return async_to_blocking(self.client_response.json(
            encoding=encoding, loads=loads, content_type=content_type))

    def __del__(self):
        self.client_response.__del__()

    def history(self) -> Tuple[aiohttp.ClientResponse, ...]:
        return self.client_response.history()

    def __repr__(self) -> str:
        return f'BlcokingClientRepsonse({repr(self.client_response)})'

    @property
    def status(self) -> int:
        return self.client_response.status

    def raise_for_status(self) -> None:
        self.client_response.raise_for_status()


class BlockingContextManager:
    def __init__(self, context_manager):
        self.context_manager = context_manager

    async def __enter__(self) -> BlockingClientResponse:
        return async_to_blocking(self.context_manager.__aenter__())

    async def __exit__(self,
                       exc_type: Optional[Type[BaseException]],
                       exc: Optional[BaseException],
                       tb: Optional[TracebackType]) -> None:
        async_to_blocking(self.context_manager.__aexit__())


class BlockingClientSession:
    def __init__(self, session: HailAsyncClientSession):
        self.session = session

    def request(self,
                method: str,
                url: aiohttp.typedefs.StrOrURL,
                **kwargs: Any) -> 'BlockingContextManager':
        return BlockingContextManager(self.session.request(
            method, url, **kwargs))

    def get(self,
            url: aiohttp.typedefs.StrOrURL,
            *,
            allow_redirects: bool = True,
            **kwargs: Any) -> 'BlockingContextManager':
        return BlockingContextManager(self.session.get(
            url, allow_redirects=allow_redirects, **kwargs))

    def options(self,
                url: aiohttp.typedefs.StrOrURL,
                *,
                allow_redirects: bool = True,
                **kwargs: Any) -> 'BlockingContextManager':
        return BlockingContextManager(self.session.options(
            url, allow_redirects=allow_redirects, **kwargs))

    def head(self,
             url: aiohttp.typedefs.StrOrURL,
             *,
             allow_redirects: bool = False,
             **kwargs: Any) -> 'BlockingContextManager':
        return BlockingContextManager(self.session.head(
            url, allow_redirects=allow_redirects, **kwargs))

    def post(self,
             url: aiohttp.typedefs.StrOrURL,
             *,
             data: Any = None, **kwargs: Any) -> 'BlockingContextManager':
        return BlockingContextManager(self.session.post(
            url, data=data, **kwargs))

    def put(self,
            url: aiohttp.typedefs.StrOrURL,
            *,
            data: Any = None,
            **kwargs: Any) -> 'BlockingContextManager':
        return BlockingContextManager(self.session.put(
            url, data=data, **kwargs))

    def patch(self,
              url: aiohttp.typedefs.StrOrURL,
              *,
              data: Any = None,
              **kwargs: Any) -> 'BlockingContextManager':
        return BlockingContextManager(self.session.patch(
            url, data=data, **kwargs))

    def delete(self,
               url: aiohttp.typedefs.StrOrURL,
               **kwargs: Any) -> 'BlockingContextManager':
        return BlockingContextManager(self.session.delete(
            url, **kwargs))

    def close(self) -> None:
        async_to_blocking(self.session.close())

    @property
    def closed(self) -> bool:
        return self.session.closed

    @property
    def cookie_jar(self) -> aiohttp.abc.AbstractCookieJar:
        return self.session.cookie_jar

    @property
    def version(self) -> Tuple[int, int]:
        return self.session.version

    def __enter__(self) -> 'BlockingClientSession':
        self.session = async_to_blocking(self.session.__aenter__())
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        self.close()


def blocking_client_session(*args, **kwargs) -> BlockingClientSession:
    return BlockingClientSession(client_session(*args, **kwargs))
