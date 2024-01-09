import aiohttp

from hailtop import httpx


class FailureInjectingClientSession(httpx.ClientSession):
    def __init__(self, should_fail):
        super().__init__()
        self.should_fail = should_fail
        self.real_session = httpx.client_session()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.real_session.close()

    def maybe_fail(self, method, path, headers):
        if self.should_fail():
            raise aiohttp.ClientResponseError(
                status=503,
                message='Service Unavailable from FailureInjectingClientSession',
                request_info=aiohttp.RequestInfo(url=path, method=method, headers=headers, real_url=path),
                history=(),
            )

    def request(self, method, url, *args, **kwargs):
        self.maybe_fail(method, url, kwargs.get('headers', {}))
        return self.real_session.request(method, url, *args, **kwargs)
