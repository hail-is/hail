import aiohttp


class FailureInjectingBatchClient:
    def __init__(self, real_client, should_fail):
        self.real_client = real_client
        self.should_fail = should_fail
        self.billing_project = real_client.billing_project

    def maybe_fail(self, method, path, headers):
        if self.should_fail():
            raise aiohttp.ClientResponseError(
                status=500,
                message='Internal Server Error from FailureInjectingBatchClient',
                request_info=aiohttp.RequestInfo(
                    url=path,
                    method=method,
                    headers=headers,
                    real_url=path),
                history=())

    async def _get(self, path, *args, **kwargs):
        self.maybe_fail('get', path, kwargs.get('headers', {}))
        return await self.real_client._get(path, *args, **kwargs)

    async def _post(self, path, *args, **kwargs):
        self.maybe_fail('post', path, kwargs.get('headers', {}))
        return await self.real_client._post(path, *args, **kwargs)

    async def _patch(self, path, *args, **kwargs):
        self.maybe_fail('patch', path, kwargs.get('headers', {}))
        return await self.real_client._patch(path, *args, **kwargs)

    async def _delete(self, path, *args, **kwargs):
        self.maybe_fail('delete', path, kwargs.get('headers', {}))
        return await self.real_client._delete(path, *args, **kwargs)
