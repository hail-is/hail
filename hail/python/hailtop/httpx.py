from typing import (Dict, Any, List, Tuple, Optional, Type, Awaitable,
                    Generator)
from types import TracebackType
import aiohttp
import socket
import logging
from hailtop.config import get_deploy_config
from .tls import internal_client_ssl_context, external_client_ssl_context
from .utils import async_to_blocking, retry_transient_errors

log = logging.getLogger('hailtop.httpx')


def client_session(*args,
                   retry_transient: bool = True,
                   raise_for_status: bool = True,
                   **kwargs) -> 'ClientSession':
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
    kwargs['raise_for_status'] = False
    return ClientSession(
        aiohttp.ClientSession(*args, **kwargs),
        retry_transient=retry_transient,
        raise_for_status=raise_for_status)


def blocking_client_session(*args, **kwargs) -> 'BlockingClientSession':
    return BlockingClientSession(client_session(*args, **kwargs))


class ResponseManager:
    def __init__(self, response_coroutine: Awaitable[aiohttp.ClientResponse]):
        self.response_coroutine = response_coroutine
        self.response: Optional[aiohttp.ClientResponse] = None

    async def __aenter__(self) -> aiohttp.ClientResponse:
        self.response = await self.response_coroutine
        return self.response

    def __await__(self) -> Generator[Any, None, aiohttp.ClientResponse]:
        assert self.response is None
        return self.response_coroutine.__await__()

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc: Optional[BaseException],
                        tb: Optional[TracebackType]) -> None:
        assert self.response is not None
        self.response.release()


class WebSocketResponseManager:
    def __init__(self, websocket_coroutine: Awaitable[aiohttp.ClientWebSocketResponse]):
        self.websocket_coroutine = websocket_coroutine
        self.websocket: Optional[aiohttp.ClientWebSocketResponse] = None

    async def __aenter__(self) -> aiohttp.ClientWebSocketResponse:
        self.websocket = await self.websocket_coroutine
        return self.websocket

    def __await__(self) -> Generator[Any, None, aiohttp.ClientWebSocketResponse]:
        assert self.websocket is None
        return self.websocket_coroutine.__await__()

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc: Optional[BaseException],
                        tb: Optional[TracebackType]) -> None:
        assert self.websocket is not None
        await self.websocket.close()


class ClientSession:
    def __init__(self, session: aiohttp.ClientSession, retry_transient: bool, raise_for_status: bool):
        self.session = session
        self.retry_transient = retry_transient
        self.raise_for_status = raise_for_status

    async def _request_and_raise_for_status(self,
                                            method: str,
                                            url: aiohttp.typedefs.StrOrURL,
                                            **kwargs: Any) -> aiohttp.ClientResponse:
        raise_for_status = kwargs.pop('raise_for_status', self.raise_for_status)
        response = await self.session._request(method, url, **kwargs)
        if raise_for_status and response.status >= 400:
            message = response.reason or ''
            try:
                message = message + '\n' + await response.text()
            except Exception as exc:
                message = message + f'\ncould not fetch text due to: {exc}'
            raise aiohttp.ClientResponseError(
                response.request_info,
                response.history,
                status=response.status,
                message=message,
                headers=response.headers)
        return response

    def ws_connect(self, url: aiohttp.typedefs.StrOrURL, **kwargs: Any):
        retry_transient = kwargs.pop('retry_transient', self.retry_transient)
        if retry_transient:
            coroutine = retry_transient_errors(self.session._ws_connect(url, **kwargs))
        else:
            coroutine = self.session._ws_connect(url, **kwargs)
        return WebSocketResponseManager(coroutine)

    def request(self,
                method: str,
                url: aiohttp.typedefs.StrOrURL,
                **kwargs: Any) -> ResponseManager:
        retry_transient = kwargs.pop('retry_transient', self.retry_transient)
        if retry_transient:
            coroutine = retry_transient_errors(
                self._request_and_raise_for_status, method, url, **kwargs)
        else:
            coroutine = self._request_and_raise_for_status(method, url, **kwargs)
        return ResponseManager(coroutine)

    def get(self,
            url: aiohttp.typedefs.StrOrURL,
            *,
            allow_redirects: bool = True,
            **kwargs: Any) -> ResponseManager:
        return self.request(
            aiohttp.hdrs.METH_GET, url, allow_redirects=allow_redirects, **kwargs)

    def options(self,
                url: aiohttp.typedefs.StrOrURL,
                *,
                allow_redirects: bool = True,
                **kwargs: Any) -> ResponseManager:
        return self.request(
            aiohttp.hdrs.METH_OPTIONS, url, allow_redirects=allow_redirects, **kwargs)

    def head(self,
             url: aiohttp.typedefs.StrOrURL,
             *,
             allow_redirects: bool = False,
             **kwargs: Any) -> ResponseManager:
        return self.request(
            aiohttp.hdrs.METH_HEAD, url, allow_redirects=allow_redirects, **kwargs)

    def post(self,
             url: aiohttp.typedefs.StrOrURL,
             *,
             data: Any = None, **kwargs: Any) -> ResponseManager:
        return self.request(
            aiohttp.hdrs.METH_POST, url, data=data, **kwargs)

    def put(self,
            url: aiohttp.typedefs.StrOrURL,
            *,
            data: Any = None,
            **kwargs: Any) -> ResponseManager:
        return self.request(
            aiohttp.hdrs.METH_PUT, url, data=data, **kwargs)

    def patch(self,
              url: aiohttp.typedefs.StrOrURL,
              *,
              data: Any = None,
              **kwargs: Any) -> ResponseManager:
        return self.request(
            aiohttp.hdrs.METH_PATCH, url, data=data, **kwargs)

    def delete(self,
               url: aiohttp.typedefs.StrOrURL,
               **kwargs: Any) -> ResponseManager:
        return self.request(
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

    async def __aenter__(self) -> 'ClientSession':
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
        self.dns = None
        self.deploy_config = get_deploy_config()

    async def resolve(self, host: str, port: int, family: int) -> List[Dict[str, Any]]:
        if self.dns is None:
            # the AsyncResolver must be created from an async function:
            # https://github.com/aio-libs/aiohttp/issues/3573#issuecomment-456742539
            self.dns = aiohttp.DefaultResolver()
        if family in (0, socket.AF_INET, socket.AF_INET6):
            maybe_address_and_port = await self.deploy_config.maybe_address(host)
            if maybe_address_and_port is not None:
                address, resolved_port = maybe_address_and_port
                return [{'hostname': host,
                         'host': address,
                         'port': resolved_port,
                         'family': family if family != 0 else socket.AF_INET,
                         'proto': 0,
                         'flags': 0}]
        return await self.dns.resolve(host, port, family)

    async def close(self) -> None:
        if self.dns is not None:
            await self.dns.close()


class BlockingClientResponse:
    def __init__(self, client_response: aiohttp.ClientResponse):
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


class BlockingWebSocketClientResponse:
    def __init__(self, ws: aiohttp.ClientWebSocketResponse):
        self.ws = ws

    @property
    def closed(self) -> bool:
        return self.ws.closed

    @property
    def close_code(self) -> Optional[int]:
        return self.ws.close_code

    @property
    def protocol(self) -> Optional[int]:
        return self.ws.protocol

    @property
    def compress(self) -> int:
        return self.ws.compress

    @property
    def client_notakeover(self) -> bool:
        return self.ws.client_notakeover

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        return self.ws.get_extra_info(name, default)

    def exception(self) -> Optional[BaseException]:
        return self.ws.exception()

    def ping(self, message: bytes = b'') -> None:
        async_to_blocking(self.ws.ping(message))

    def pong(self, message: bytes = b'') -> None:
        async_to_blocking(self.ws.pong(message))

    def send_str(self, data: str,
                 compress: Optional[int] = None) -> None:
        return async_to_blocking(self.ws.send_str(data, compress))

    def send_bytes(self, data: bytes,
                   compress: Optional[int] = None) -> None:
        return async_to_blocking(self.ws.send_bytes(data, compress))

    def send_json(self, data: Any,
                  compress: Optional[int] = None,
                  *, dumps: aiohttp.typedefs.JSONEncoder = aiohttp.typedefs.DEFAULT_JSON_ENCODER) -> None:
        return async_to_blocking(self.ws.send_json(data, compress, dumps=dumps))

    def close(self, *, code: int = 1000, message: bytes = b'') -> bool:
        return async_to_blocking(self.ws.close(code=code, message=message))

    def receive(self, timeout: Optional[float] = None) -> aiohttp.WSMessage:
        return async_to_blocking(self.ws.receive(timeout))

    def receive_str(self, *, timeout: Optional[float] = None) -> str:
        return async_to_blocking(self.ws.receive_str(timeout))

    def receive_bytes(self, *, timeout: Optional[float] = None) -> bytes:
        return async_to_blocking(self.ws.receive_bytes(timeout))

    def receive_json(self,
                     *, loads: aiohttp.typedefs.JSONDecoder = aiohttp.typedefs.DEFAULT_JSON_DECODER,
                     timeout: Optional[float] = None) -> Any:
        return async_to_blocking(self.ws.receive_json(loads=loads, timeout=timeout))

    def __iter__(self) -> 'BlockingWebSocketClientResponse':
        return self

    def __next__(self) -> aiohttp.WSMessage:
        try:
            return async_to_blocking(self.ws.__anext__())
        except StopAsyncIteration as exc:
            raise StopIteration() from exc


class BlockingContextManager:
    def __init__(self, context_manager):
        self.context_manager = context_manager

    def __enter__(self) -> BlockingClientResponse:
        return BlockingClientResponse(async_to_blocking(self.context_manager.__aenter__()))

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc: Optional[BaseException],
                 tb: Optional[TracebackType]) -> None:
        async_to_blocking(self.context_manager.__aexit__(exc_type, exc, tb))


class BlockingWebSocketContextManager:
    def __init__(self, context_manager):
        self.context_manager = context_manager

    def __enter__(self) -> BlockingWebSocketClientResponse:
        return BlockingWebSocketClientResponse(async_to_blocking(self.context_manager.__aenter__()))

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc: Optional[BaseException],
                 tb: Optional[TracebackType]) -> None:
        async_to_blocking(self.context_manager.__aexit__(exc_type, exc, tb))


class BlockingClientSession:
    def __init__(self, session: ClientSession):
        self.session = session

    def request(self,
                method: str,
                url: aiohttp.typedefs.StrOrURL,
                **kwargs: Any) -> 'BlockingContextManager':
        return BlockingContextManager(self.session.request(
            method, url, **kwargs))

    def ws_connect(self,
                   url: aiohttp.typedefs.StrOrURL,
                   **kwargs: Any) -> 'BlockingWebSocketContextManager':
        return BlockingWebSocketContextManager(self.session.ws_connect(
            url, **kwargs))

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
