from typing import Any, Tuple, Optional, Type, TypeVar, Generic, Callable, Union
import asyncio
from types import TracebackType
import orjson
import aiohttp
import aiohttp.abc
import aiohttp.typedefs

from .utils import async_to_blocking
from .tls import internal_client_ssl_context, external_client_ssl_context
from .config.deploy_config import get_deploy_config


class ClientResponseError(aiohttp.ClientResponseError):
    def __init__(self,
                 request_info: aiohttp.RequestInfo,
                 history: Tuple[aiohttp.ClientResponse, ...],
                 body: str = "",
                 **kwargs):
        super().__init__(request_info, history, **kwargs)
        self.body = body

    def __str__(self) -> str:
        return (f"{self.status}, message={self.message!r}, "
                f"url={self.request_info.real_url!r} body={self.body!r}")

    def __repr__(self) -> str:
        args = f"{self.request_info!r}, {self.history!r}"
        if self.status != 0:
            args += f", status={self.status!r}"
        if self.message != "":
            args += f", message={self.message!r}"
        if self.headers is not None:
            args += f", headers={self.headers!r}"
        if self.body is not None:
            args += f", body={self.body!r}"
        return f"{type(self).__name__}({args})"


class ClientResponse:
    def __init__(self, client_response: aiohttp.ClientResponse):
        self.client_response = client_response

    async def release(self) -> None:
        return await self.client_response.release()

    @property
    def closed(self) -> bool:
        return self.client_response.closed

    def close(self) -> None:
        return self.client_response.close()

    async def wait_for_close(self) -> None:
        return await self.wait_for_close()

    async def read(self) -> bytes:
        return await self.client_response.read()

    def get_encoding(self) -> str:
        return self.client_response.get_encoding()

    async def text(self, encoding: Optional[str] = None, errors: str = 'strict'):
        return await self.client_response.text(encoding=encoding, errors=errors)

    async def json(self):
        encoding = self.get_encoding()

        if encoding != 'utf-8':
            return await self.client_response.json()

        content_type = self.client_response.headers.get(aiohttp.hdrs.CONTENT_TYPE, None)
        assert content_type is None or content_type == 'application/json', self.client_response
        return orjson.loads(await self.read())

    async def __aenter__(self) -> "ClientResponse":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.release()


class ClientSession:
    def __init__(self,
                 *args,
                 raise_for_status: bool = True,
                 timeout: Union[aiohttp.ClientTimeout, float, None] = None,
                 **kwargs):
        try:
            self.loop_at_creation_time = asyncio.get_running_loop()
        except RuntimeError as err:
            raise ValueError(
                'aiohttp.ClientSession stashes a copy of the event loop into a field. Very confusing '
                'errors arise if that stashed event loop differs from the currently running one. '
                'We refuse to create an httpx.ClientSession outside of an event loop for this reason. '
                'Take care to ensure the event loop you use when you create the ClientSession is the '
                f'same as the running event loop when you make a request. {self.loop_at_creation_time} '
                f'{asyncio.get_running_loop()}'
            )

        location = get_deploy_config().location()
        if location == 'external':
            tls = external_client_ssl_context()
        elif location == 'k8s':
            tls = internal_client_ssl_context()
        else:
            assert location in ('gce', 'azure')
            # no encryption on the internal gateway
            tls = external_client_ssl_context()

        assert 'connector' not in kwargs

        if timeout is None:
            timeout = aiohttp.ClientTimeout(total=5)

        self.raise_for_status = raise_for_status
        self.client_session = aiohttp.ClientSession(
            *args,
            timeout=timeout,
            raise_for_status=False,
            connector=aiohttp.TCPConnector(ssl=tls),
            **kwargs
        )

    def request(
        self, method: str, url: aiohttp.typedefs.StrOrURL, **kwargs: Any
    ) -> aiohttp.client._RequestContextManager:
        raise_for_status = kwargs.pop('raise_for_status', self.raise_for_status)

        async def request_and_raise_for_status():
            if self.loop_at_creation_time != asyncio.get_running_loop():
                raise ValueError(
                    f'aiohttp.ClientSession will raise confusing errors if the running event loop at '
                    f'creation time differs from the running event loop at request time. {self.loop_at_creation_time} '
                    f'{asyncio.get_running_loop()}'
                )

            json_data = kwargs.pop('json', None)
            if json_data is not None:
                if kwargs.get('data') is not None:
                    raise ValueError(
                        'data and json parameters cannot be used at the same time')
                kwargs['data'] = aiohttp.BytesPayload(
                    value=orjson.dumps(json_data),
                    # https://github.com/ijl/orjson#serialize
                    #
                    # "The output is a bytes object containing UTF-8"
                    encoding="utf-8",
                    content_type="application/json",
                )
            resp = await self.client_session._request(method, url, **kwargs)
            if raise_for_status:
                if resp.status >= 400:
                    # reason should always be not None for a started response
                    assert resp.reason is not None
                    body = (await resp.read()).decode()
                    await resp.release()
                    raise ClientResponseError(
                        resp.request_info,
                        resp.history,
                        status=resp.status,
                        message=resp.reason,
                        headers=resp.headers,
                        body=body
                    )
            return resp
        return aiohttp.client._RequestContextManager(request_and_raise_for_status())

    def ws_connect(
        self, *args, **kwargs
    ) -> aiohttp.client._WSRequestContextManager:
        if self.loop_at_creation_time != asyncio.get_running_loop():
            raise ValueError(
                f'aiohttp.ClientSession will raise confusing errors if the running event loop at '
                f'creation time differs from the running event loop at request time. {self.loop_at_creation_time} '
                f'{asyncio.get_running_loop()}'
            )
        return self.client_session.ws_connect(*args, **kwargs)

    def get(
        self, url: aiohttp.typedefs.StrOrURL, *, allow_redirects: bool = True, **kwargs: Any
    ) -> aiohttp.client._RequestContextManager:
        return self.request('GET', url, allow_redirects=allow_redirects, **kwargs)

    async def get_read_json(
        self, *args, **kwargs
    ) -> Any:
        async with self.get(*args, **kwargs) as resp:
            return await resp.json()

    async def get_read(
        self, *args, **kwargs
    ) -> bytes:
        async with self.get(*args, **kwargs) as resp:
            return await resp.read()

    def options(
        self, url: aiohttp.typedefs.StrOrURL, *, allow_redirects: bool = True, **kwargs: Any
    ) -> aiohttp.client._RequestContextManager:
        return self.request('OPTIONS', url, allow_redirects=allow_redirects, **kwargs)

    def head(
        self, url: aiohttp.typedefs.StrOrURL, *, allow_redirects: bool = False, **kwargs: Any
    ) -> aiohttp.client._RequestContextManager:
        return self.request('HEAD', url, allow_redirects=allow_redirects, **kwargs)

    def post(
        self, url: aiohttp.typedefs.StrOrURL, *, data: Any = None, **kwargs: Any
    ) -> aiohttp.client._RequestContextManager:
        return self.request('POST', url, data=data, **kwargs)

    async def post_read_json(
        self, *args, **kwargs
    ) -> Any:
        async with self.post(*args, **kwargs) as resp:
            return await resp.json()

    async def post_read(
        self, *args, **kwargs
    ) -> bytes:
        async with self.post(*args, **kwargs) as resp:
            return await resp.read()

    def put(
        self, url: aiohttp.typedefs.StrOrURL, *, data: Any = None, **kwargs: Any
    ) -> aiohttp.client._RequestContextManager:
        return self.request('PUT', url, data=data, **kwargs)

    def patch(
        self, url: aiohttp.typedefs.StrOrURL, *, data: Any = None, **kwargs: Any
    ) -> aiohttp.client._RequestContextManager:
        return self.request('PATCH', url, data=data, **kwargs)

    def delete(
        self, url: aiohttp.typedefs.StrOrURL, **kwargs: Any
    ) -> aiohttp.client._RequestContextManager:
        return self.request('DELETE', url, **kwargs)

    async def close(self) -> None:
        await self.client_session.close()
        # - Following warning mitigation described here: https://github.com/aio-libs/aiohttp/pull/2045
        # - Fixed in aiohttp 4.0.0: https://github.com/aio-libs/aiohttp/issues/1925
        await asyncio.sleep(0.250)

    @property
    def closed(self) -> bool:
        return self.client_session.closed

    @property
    def cookie_jar(self) -> aiohttp.abc.AbstractCookieJar:
        return self.client_session.cookie_jar

    @property
    def version(self) -> Tuple[int, int]:
        return self.client_session.version

    async def __aenter__(self) -> "ClientSession":
        if self.loop_at_creation_time != asyncio.get_running_loop():
            raise ValueError(
                'aiohttp.ClientSession will raise confusing errors if the running event loop at '
                'creation time differs from the running event loop at request time.'
            )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.client_session.__aexit__(exc_type, exc_val, exc_tb)


def client_session(*args, **kwargs) -> ClientSession:
    return ClientSession(*args, **kwargs)


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
             encoding: Optional[str] = None,
             loads: aiohttp.typedefs.JSONDecoder = aiohttp.typedefs.DEFAULT_JSON_DECODER,
             content_type: Optional[str] = 'application/json') -> Any:
        return async_to_blocking(self.client_response.json(
            encoding=encoding, loads=loads, content_type=content_type))

    def __del__(self):
        self.client_response.__del__()

    def history(self) -> Tuple[aiohttp.ClientResponse, ...]:
        return self.client_response.history

    def __repr__(self) -> str:
        return f'BlockingClientRepsonse({self.client_response!r})'

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
    def __init__(self, session: ClientSession):
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
