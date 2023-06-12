from typing import Optional

import aiohttp

from hailtop import httpx
from hailtop.utils import retry_transient_errors


class GoogleMetadataServerClient:
    def __init__(self, http_session: httpx.ClientSession):
        self._session = http_session
        self._numeric_project_id: Optional[str] = None

    async def numeric_project_id(self):
        if self._numeric_project_id is None:
            self._numeric_project_id = await self._request('/project/numeric-project-id')
        return self._numeric_project_id

    async def _request(self, path: str) -> str:
        return await retry_transient_errors(
            self._session.get_read_json,
            f'http://metadata.google.internal/computeMetadata/v1{path}',
            headers={'Metadata-Flavor': 'Google'},
            timeout=aiohttp.ClientTimeout(total=60),  # type: ignore
        )
