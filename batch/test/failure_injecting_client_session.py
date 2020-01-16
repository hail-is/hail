import aiohttp


class FailureInjectingClientSession:
    def __init__(self, real_session, should_fail):
        self.real_session = real_session
        self.should_fail = should_fail
        self.billing_project = real_session.billing_project

    def maybe_fail(self, method, path, headers):
        if self.should_fail():
            raise aiohttp.ClientResponseError(
                status=500,
                message='Internal Server Error from FailureInjectingClientSession',
                request_info=aiohttp.RequestInfo(
                    url=path,
                    method=method,
                    headers=headers,
                    real_url=path),
                history=())

    async def request(self, method, path, *args, **kwargs):
        self.maybe_fail(method, path, kwargs.get('headers', {}))
        return await self.real_session.request(method, path, *args, **kwargs)

    async def close(self, *args, **kwargs):
        return await self.real_session.close()
