from typing import Any, Tuple, Optional, Type, TypeVar, Generic, Callable, Union
from types import TracebackType
import aiohttp

from .utils import async_to_blocking
from .tls import internal_client_ssl_context, external_client_ssl_context
from .config.deploy_config import get_deploy_config


def client_session(*args,
                   raise_for_status: bool = True,
                   timeout: Union[aiohttp.ClientTimeout, float] = None,
                   **kwargs) -> aiohttp.ClientSession:
    location = get_deploy_config().location()
    if location == 'external':
        tls = external_client_ssl_context()
    elif location == 'k8s':
        tls = internal_client_ssl_context()
    else:
        assert location == 'gce'
        # no encryption on the internal gateway
        tls = external_client_ssl_context()

    assert 'connector' not in kwargs
    kwargs['connector'] = aiohttp.TCPConnector(ssl=tls)

    kwargs['raise_for_status'] = raise_for_status

    if timeout is None:
        timeout = aiohttp.ClientTimeout(total=5)
    kwargs['timeout'] = timeout

    return aiohttp.ClientSession(*args, **kwargs)


def blocking_client_session(*args, **kwargs) -> 'BlockingClientSession':
    return BlockingClientSession(client_session(*args, **kwargs))


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
        return self.client_response.history

    def __repr__(self) -> str:
        return f'BlockingClientRepsonse({repr(self.client_response)})'

    @property
    def status(self) -> int:
        return self.client_response.status

    def raise_for_status(self) -> None:
        self.client_response.raise_for_status()


class BlockingClientWebSocketResponse:
    def __init__(self, ws: aiohttp.ClientWebSocketResponse):
        self.ws = ws

    @property
    def closed(self) -> bool:
        return self.ws.closed

    @property
    def close_code(self) -> Optional[int]:
        return self.ws.close_code

    @property
    def protocol(self) -> Optional[str]:
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
        return async_to_blocking(self.ws.receive_str(timeout=timeout))

    def receive_bytes(self, *, timeout: Optional[float] = None) -> bytes:
        return async_to_blocking(self.ws.receive_bytes(timeout=timeout))

    def receive_json(self,
                     *, loads: aiohttp.typedefs.JSONDecoder = aiohttp.typedefs.DEFAULT_JSON_DECODER,
                     timeout: Optional[float] = None) -> Any:
        return async_to_blocking(self.ws.receive_json(loads=loads, timeout=timeout))

    def __iter__(self) -> 'BlockingClientWebSocketResponse':
        return self

    def __next__(self) -> aiohttp.WSMessage:
        try:
            return async_to_blocking(self.ws.__anext__())
        except StopAsyncIteration as exc:
            raise StopIteration() from exc


T = TypeVar('T')  # pylint: disable=invalid-name
U = TypeVar('U')  # pylint: disable=invalid-name


class AsyncToBlockingContextManager(Generic[T, U]):
    def __init__(self, context_manager, wrap: Callable[[T], U]):
        self.context_manager = context_manager
        self.wrap = wrap

    def __enter__(self) -> U:
        return self.wrap(async_to_blocking(self.context_manager.__aenter__()))

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc: Optional[BaseException],
                 tb: Optional[TracebackType]) -> None:
        async_to_blocking(self.context_manager.__aexit__(exc_type, exc, tb))


class BlockingClientResponseContextManager(AsyncToBlockingContextManager):
    def __init__(self, context_manager):
        super().__init__(context_manager, BlockingClientResponse)


class BlockingClientWebSocketResponseContextManager(AsyncToBlockingContextManager):
    def __init__(self, context_manager):
        super().__init__(context_manager, BlockingClientWebSocketResponse)


class BlockingClientSession:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    def request(self,
                method: str,
                url: aiohttp.typedefs.StrOrURL,
                **kwargs: Any) -> BlockingClientResponseContextManager:
        return BlockingClientResponseContextManager(
            self.session.request(method, url, **kwargs))

    def ws_connect(self,
                   url: aiohttp.typedefs.StrOrURL,
                   **kwargs: Any) -> BlockingClientWebSocketResponseContextManager:
        return BlockingClientWebSocketResponseContextManager(
            self.session.ws_connect(url, **kwargs))

    def get(self,
            url: aiohttp.typedefs.StrOrURL,
            *,
            allow_redirects: bool = True,
            **kwargs: Any) -> BlockingClientResponseContextManager:
        return BlockingClientResponseContextManager(
            self.session.get(url, allow_redirects=allow_redirects, **kwargs))

    def options(self,
                url: aiohttp.typedefs.StrOrURL,
                *,
                allow_redirects: bool = True,
                **kwargs: Any) -> BlockingClientResponseContextManager:
        return BlockingClientResponseContextManager(
            self.session.options(url, allow_redirects=allow_redirects, **kwargs))

    def head(self,
             url: aiohttp.typedefs.StrOrURL,
             *,
             allow_redirects: bool = False,
             **kwargs: Any) -> BlockingClientResponseContextManager:
        return BlockingClientResponseContextManager(self.session.head(
            url, allow_redirects=allow_redirects, **kwargs))

    def post(self,
             url: aiohttp.typedefs.StrOrURL,
             *,
             data: Any = None, **kwargs: Any) -> BlockingClientResponseContextManager:
        return BlockingClientResponseContextManager(self.session.post(
            url, data=data, **kwargs))

    def put(self,
            url: aiohttp.typedefs.StrOrURL,
            *,
            data: Any = None,
            **kwargs: Any) -> BlockingClientResponseContextManager:
        return BlockingClientResponseContextManager(self.session.put(
            url, data=data, **kwargs))

    def patch(self,
              url: aiohttp.typedefs.StrOrURL,
              *,
              data: Any = None,
              **kwargs: Any) -> BlockingClientResponseContextManager:
        return BlockingClientResponseContextManager(self.session.patch(
            url, data=data, **kwargs))

    def delete(self,
               url: aiohttp.typedefs.StrOrURL,
               **kwargs: Any) -> BlockingClientResponseContextManager:
        return BlockingClientResponseContextManager(self.session.delete(
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
