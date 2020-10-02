from typing import Dict, Any, List, Union, Tuple, Optional, Type, TracebackType
import aiohttp
import socket
import logging
from urllib3.poolmanager import PoolManager
from hailtop.config import get_deploy_config
from .tls import (get_in_cluster_client_ssl_context,
                  get_context_specific_client_ssl_context,
                  _get_ssl_config)
from .utils import async_to_blocking

log = logging.getLogger('hailtop.httpx')


def in_cluster_ssl_client_session(*args, **kwargs) -> aiohttp.ClientSession:
    assert 'connector' not in kwargs
    kwargs['connector'] = aiohttp.TCPConnector(
        ssl=get_in_cluster_client_ssl_context(),
        resolver=HailResolver())
    return aiohttp.ClientSession(*args, **kwargs)


def get_context_specific_ssl_client_session(*args, **kwargs) -> aiohttp.ClientSession:
    assert 'connector' not in kwargs
    kwargs['connector'] = aiohttp.TCPConnector(
        ssl=get_context_specific_client_ssl_context(),
        resolver=HailResolver())
    return aiohttp.ClientSession(*args, **kwargs)


class HailResolver(aiohttp.abc.AbstractResolver):
    """Use Hail in-cluster DNS with fallback."""
    def __init__(self):
        self.dns = aiohttp.AsyncResolver()
        self.deploy_config = get_deploy_config()

    async def resolve(self, host: str, port: int, family: int) -> List[Dict[str, Any]]:
        if family in (socket.AF_INET, socket.AF_INET6):
            maybe_address_and_port = self.deploy_config.maybe_address(host)
            if maybe_address_and_port is not None:
                address, resolved_port = maybe_address_and_port
                return [{'hostname': host,
                         'host': address,
                         'port': resolved_port,
                         'family': family,
                         'proto': 0,
                         'flags': 0}]
        return self.dns.resolve(host, port, family)

    async def close(self) -> None:
        self.dns.close()


class BlockingClientResponse:
    def __init__(self, client_response):
        self.client_response = client_response

    def read(self) -> bytes:
        return async_to_blocking(self.client_response.read())

    def text(self, encoding: Optional[str] = None, errors: str = 'strict') -> str:
        return async_to_blocking(self.client_response.text())

    def json(self, *,
             encoding: str = None,
             loads: aiohttp.typedefs.JSONDecoder = aiohttp.typedefs.DEFAULT_JSON_DECODER,
             content_type: Optional[str] = 'application/json') -> Any:
        return async_to_blocking(self.client_response.json())

    def __del__(self) -> str:
        self.client_response.__del__()

    def history(self) -> Tuple[aiohttp.ClientResponse, ...]:
        return self.client_response.history()

    def __repr__(self) -> str:
        return f'BlcokingClientRepsonse({repr(self.client_response)})'

    def raise_for_status(self) -> None:
        self.client_response.raise_for_status()


class BlockingContextManager:
    def __init__(self, context_manager):
        self.context_manager = context_manager

    async def __enter__(self) -> BlockingClientResponse:
        return async_to_blocking(self.context_manager.__aenter__)

    async def __exit__(self,
                       exc_type: Optional[Type[BaseException]],
                       exc: Optional[BaseException],
                       tb: Optional[TracebackType]) -> None:
        async_to_blocking(self.context_manager.__aexit__)


class BlockingClientSession:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    def request(self,
                method: str,
                url: aiohttp.typedefs.StrOrURL,
                **kwargs: Any) -> 'BlockingContextManager':
        return BlockingContextManager(self.session.request(
            self, method, url, **kwargs))

    def get(self,
            url: aiohttp.typedefs.StrOrURL,
            *,
            allow_redirects: bool = True,
            **kwargs: Any) -> 'BlockingContextManager':
        return BlockingContextManager(self.session.get(
            self, url, allow_redirects=allow_redirects, **kwargs))

    def options(self,
                url: aiohttp.typedefs.StrOrURL,
                *,
                allow_redirects: bool = True,
                **kwargs: Any) -> 'BlockingContextManager':
        return BlockingContextManager(self.session.options(
            self, url, allow_redirects=allow_redirects, **kwargs))

    def head(self,
             url: aiohttp.typedefs.StrOrURL,
             *,
             allow_redirects: bool = False,
             **kwargs: Any) -> 'BlockingContextManager':
        return BlockingContextManager(self.session.head(
            self, url, allow_redirects=allow_redirects, **kwargs))

    def post(self,
             url: aiohttp.typedefs.StrOrURL,
             *,
             data: Any = None, **kwargs: Any) -> 'BlockingContextManager':
        return BlockingContextManager(self.session.post(
            self, url, **kwargs))

    def put(self,
            url: aiohttp.typedefs.StrOrURL,
            *,
            data: Any = None,
            **kwargs: Any) -> 'BlockingContextManager':
        return BlockingContextManager(self.session.put(
            self, url, data=data, **kwargs))

    def patch(self,
              url: aiohttp.typedefs.StrOrURL,
              *,
              data: Any = None,
              **kwargs: Any) -> 'BlockingContextManager':
        return BlockingContextManager(self.session.patch(
            self, url, data=data, **kwargs))

    def delete(self,
               url: aiohttp.typedefs.StrOrURL,
               **kwargs: Any) -> 'BlockingContextManager':
        return BlockingContextManager(self.session.delete(self, url, **kwargs))

    def close(self) -> None:
        async_to_blocking(self.session.close())

    @property
    def closed(self) -> bool:
        return self.session.closed()

    @property
    def cookie_jar(self) -> aiohttp.AbstractCookieJar:
        return self.session.cookie_jar

    @property
    def version(self) -> Tuple[int, int]:
        return self.session.version

    def __enter__(self) -> 'BlockingClientSession':
        self.session = async_to_blocking(self.session.__aenter__)
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        self.close()


def blocking_in_cluster_ssl_client_session(*args, **kwargs) -> BlockingClientSession:
    async_session = in_cluster_ssl_client_session(*args, **kwargs)
    return BlockingClientSession(async_session)
