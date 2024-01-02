from typing import Optional, AsyncGenerator, Any, List, Mapping, Union

import aiohttp
from hailtop.utils import RateLimit, sleep_before_try, url_and_params

from ..credentials import AzureCredentials
from ...common import CloudBaseClient
from ...common.session import BaseSession, Session
from ...common.credentials import AnonymousCloudCredentials


class AzureBaseClient(CloudBaseClient):
    def __init__(self,
                 base_url: str,
                 *,
                 session: Optional[BaseSession] = None,
                 rate_limit: Optional[RateLimit] = None,
                 credentials: Optional[Union['AzureCredentials', AnonymousCloudCredentials]] = None,
                 credentials_file: Optional[str] = None,
                 scopes: Optional[List[str]] = None,
                 params: Optional[Mapping[str, str]] = None,
                 **kwargs):
        if session is None:
            session = Session(
                credentials=credentials or AzureCredentials.from_file_or_default(credentials_file, scopes),
                params=params,
                **kwargs
            )
        elif credentials_file is not None or credentials is not None:
            raise ValueError('Do not provide credentials_file or credentials when session is not None')

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

    async def delete_and_wait(self, path: Optional[str] = None, *, url: Optional[str] = None, **kwargs) -> aiohttp.ClientResponse:
        tries = 1
        while True:
            resp = await self.delete(path, url=url, **kwargs)
            if resp.status == 204:
                return resp
            tries += 1
            await sleep_before_try(tries, base_delay_ms=5_000)
