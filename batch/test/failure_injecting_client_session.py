import aiohttp

from hailtop.utils import async_to_blocking
from hailtop.httpx import client_session


class FailureInjectingClientSession:
    def __init__(self, should_fail):
        self.should_fail = should_fail
        self.real_session = client_session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        async_to_blocking(self.real_session.close())

    def maybe_fail(self, method, path, headers):
        if self.should_fail():
            raise aiohttp.ClientResponseError(
                status=503,
                message='Service Unavailable from FailureInjectingClientSession',
                request_info=aiohttp.RequestInfo(url=path, method=method, headers=headers, real_url=path),
                history=(),
            )

    async def request(self, method, path, *args, **kwargs):
        self.maybe_fail(method, path, kwargs.get('headers', {}))
        return await self.real_session.request(method, path, *args, **kwargs)
