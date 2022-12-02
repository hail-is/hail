from typing import Optional, AsyncGenerator, Any

import aiohttp
from hailtop.utils import RateLimit, sleep_and_backoff, url_and_params

from ...common import CloudBaseClient
from ..session import AzureSession


class AzureBaseClient(CloudBaseClient):
    _session: AzureSession

    def __init__(self, base_url: str, *, session: Optional[AzureSession] = None,
                 rate_limit: Optional[RateLimit] = None, **kwargs):
        if session is None:
            session = AzureSession(**kwargs)
        super().__init__(base_url, session, rate_limit=rate_limit)

    async def _paged_get(self, path, **kwargs) -> AsyncGenerator[Any, None]:
        page = await self.get(path, **kwargs)
        for v in page.get('value', []):
            yield v
        next_link = page.get('nextLink')
        while next_link is not None:
            url, params = url_and_params(next_link)
            page = await self.get(url=url, params=params)
            for v in page['value']:
                yield v
            next_link = page.get('nextLink')

    async def delete(self, path: Optional[str] = None, *, url: Optional[str] = None, **kwargs) -> aiohttp.ClientResponse:
        if url is None:
            assert path
            url = f'{self._base_url}{path}'
        async with await self._session.delete(url, **kwargs) as resp:
            return resp

    async def delete_and_wait(self, path: Optional[str] = None, *, url: Optional[str] = None, **kwargs) -> aiohttp.ClientResponse:
        delay = 5
        while True:
            resp = await self.delete(path, url=url, **kwargs)
            if resp.status == 204:
                return resp
            delay = await sleep_and_backoff(delay)
