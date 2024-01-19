from typing import Optional

import aiohttp

from hailtop import httpx
from hailtop.utils import retry_transient_errors


class GoogleMetadataServerClient:
    def __init__(self, http_session: httpx.ClientSession):
        self._session = http_session
        self._project_id: Optional[str] = None
        self._numeric_project_id: Optional[str] = None

    async def project(self) -> str:
        if self._project_id is None:
            self._project_id = await retry_transient_errors(self._get_text, '/project/project-id')
        return self._project_id

    async def numeric_project_id(self) -> str:
        if self._numeric_project_id is None:
            self._numeric_project_id = await retry_transient_errors(self._get_text, '/project/numeric-project-id')
        return self._numeric_project_id

    async def _get_text(self, path: str) -> str:
        url = f'http://metadata.google.internal/computeMetadata/v1{path}'
        headers = {'Metadata-Flavor': 'Google'}
        timeout = aiohttp.ClientTimeout(total=60)
        res = await self._session.get_read(url, headers=headers, timeout=timeout)
        return res.decode('utf-8')
