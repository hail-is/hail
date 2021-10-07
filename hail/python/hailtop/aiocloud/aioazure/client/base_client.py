from typing import Optional

import aiohttp
from hailtop.utils import RateLimit, sleep_and_backoff

from ...common import CloudBaseClient
from ..session import AzureSession


class AzureBaseClient(CloudBaseClient):
    _session: AzureSession

    def __init__(self, base_url: str, *, session: Optional[AzureSession] = None,
                 rate_limit: RateLimit = None, **kwargs):
        if session is None:
            session = AzureSession(**kwargs)
        super().__init__(base_url, session, rate_limit=rate_limit)

    async def delete(self, path: str, **kwargs) -> aiohttp.ClientResponse:
        async with await self._session.delete(
                f'{self._base_url}{path}', **kwargs) as resp:
            return resp

    async def delete_and_wait(self, path: str, **kwargs) -> aiohttp.ClientResponse:
        delay = 5
        while True:
            resp = await self.delete(path, **kwargs)
            if resp.status == 204:
                return resp
            delay = await sleep_and_backoff(delay)
