import asyncio
from types import TracebackType
from typing import Any, Optional, Tuple, Type, Union

import aiohttp
import aiohttp.abc
import aiohttp.typedefs
import orjson

from .config import ConfigVariable, configuration_of
from .config.deploy_config import get_deploy_config


class ClientResponseError(aiohttp.ClientResponseError):
    def __init__(
        self, request_info: aiohttp.RequestInfo, history: Tuple[aiohttp.ClientResponse, ...], body: str = "", **kwargs
    ):
        super().__init__(request_info, history, **kwargs)
        self.body = body

    def __str__(self) -> str:
        return f"{self.status}, message={self.message!r}, " f"url={self.request_info.real_url!r} body={self.body!r}"

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
    def __init__(
        self,
        *args,
        raise_for_status: bool = True,
        timeout: Union[aiohttp.ClientTimeout, float, int, None] = None,
        **kwargs,
    ):
        tls = get_deploy_config().client_ssl_context()
        assert 'connector' not in kwargs

        configuration_of_timeout = configuration_of(ConfigVariable.HTTP_TIMEOUT_IN_SECONDS, timeout, 5)
        del timeout

        if isinstance(configuration_of_timeout, str):
            configuration_of_timeout = float(configuration_of_timeout)
        if isinstance(configuration_of_timeout, (float, int)):
            configuration_of_timeout = aiohttp.ClientTimeout(total=configuration_of_timeout)
        assert isinstance(configuration_of_timeout, aiohttp.ClientTimeout)

        self.loop = asyncio.get_running_loop()
        self.raise_for_status = raise_for_status
        self.client_session = aiohttp.ClientSession(
            *args,
            timeout=configuration_of_timeout,
            raise_for_status=False,
            connector=aiohttp.TCPConnector(ssl=tls),
            **kwargs,
        )

    def request(
        self, method: str, url: aiohttp.typedefs.StrOrURL, **kwargs: Any
    ) -> aiohttp.client._RequestContextManager:
        if self.loop != asyncio.get_running_loop():
            raise ValueError(
                f'ClientSession must be created and used in same loop {self.loop} != {asyncio.get_running_loop()}.'
            )
        raise_for_status = kwargs.pop('raise_for_status', self.raise_for_status)

        timeout = kwargs.get('timeout')
        if timeout and isinstance(timeout, (float, int)):
            kwargs['timeout'] = aiohttp.ClientTimeout(total=timeout)

        async def request_and_raise_for_status():
            json_data = kwargs.pop('json', None)
            if json_data is not None:
                if kwargs.get('data') is not None:
                    raise ValueError('data and json parameters cannot be used at the same time')
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
                        body=body,
                    )
            return resp

        return aiohttp.client._RequestContextManager(request_and_raise_for_status())

    def ws_connect(self, *args, **kwargs) -> aiohttp.client._WSRequestContextManager:
        return self.client_session.ws_connect(*args, **kwargs)

    def get(
        self, url: aiohttp.typedefs.StrOrURL, *, allow_redirects: bool = True, **kwargs: Any
    ) -> aiohttp.client._RequestContextManager:
        return self.request('GET', url, allow_redirects=allow_redirects, **kwargs)

    async def get_read_json(self, *args, **kwargs) -> Any:
        async with self.get(*args, **kwargs) as resp:
            return await resp.json()

    async def get_read(self, *args, **kwargs) -> bytes:
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

    async def post_read_json(self, *args, **kwargs) -> Any:
        async with self.post(*args, **kwargs) as resp:
            return await resp.json()

    async def post_read(self, *args, **kwargs) -> bytes:
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

    def delete(self, url: aiohttp.typedefs.StrOrURL, **kwargs: Any) -> aiohttp.client._RequestContextManager:
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
