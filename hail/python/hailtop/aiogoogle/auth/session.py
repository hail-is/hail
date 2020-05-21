import aiohttp
from hailtop.utils import request_retry_transient_errors
from .credentials import Credentials
from .access_token import AccessToken


class Session:
    def __init__(self, *, credentials=None, **kwargs):
        if credentials is None:
            credentials = Credentials.default_credentials()

        if 'raise_for_status' not in kwargs:
            kwargs['raise_for_status'] = True
        self._session = aiohttp.ClientSession(**kwargs)
        self._access_token = AccessToken(credentials)

    async def request(self, method, url, **kwargs):
        auth_headers = await self._access_token.auth_headers(self._session)
        if 'headers' in kwargs:
            kwargs['headers'].update(auth_headers)
        else:
            kwargs['headers'] = auth_headers
        return await request_retry_transient_errors(self._session, method, url, **kwargs)

    async def get(self, url, **kwargs):
        return await self.request('GET', url, **kwargs)

    async def post(self, url, **kwargs):
        return await self.request('POST', url, **kwargs)

    async def put(self, url, **kwargs):
        return await self.request('PUT', url, **kwargs)

    async def delete(self, url, **kwargs):
        return await self.request('DELETE', url, **kwargs)

    async def head(self, url, **kwargs):
        return await self.request('HEAD', url, **kwargs)

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._access_token = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
