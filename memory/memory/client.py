import aiohttp

from hailtop.aiocloud.aiogoogle import GoogleStorageAsyncFS
from hailtop.auth import hail_credentials
from hailtop.config import get_deploy_config
from hailtop.httpx import client_session
from hailtop.utils import retry_transient_errors


class MemoryClient:
    def __init__(self, gcs_project=None, fs=None, deploy_config=None, session=None, headers=None):
        if not deploy_config:
            self._deploy_config = get_deploy_config()
        else:
            self._deploy_config = deploy_config

        self.url = self._deploy_config.base_url('memory')
        self.objects_url = f'{self.url}/api/v1alpha/objects'
        self._session = session

        if fs is None:
            fs = GoogleStorageAsyncFS(project=gcs_project)
        self._fs = fs

        self._headers = {}
        if headers:
            self._headers.update(headers)

    async def async_init(self):
        if self._session is None:
            self._session = client_session()
        self._headers.update(await hail_credentials().auth_headers())

    async def _get_file_if_exists(self, filename):
        params = {'q': filename}
        try:
            return await retry_transient_errors(
                self._session.get_return_json, self.objects_url, params=params, headers=self._headers
            )
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                return None
            raise e

    async def read_file(self, filename):
        data = await self._get_file_if_exists(filename)
        if data is not None:
            return data
        return await self._fs.read(filename)

    async def write_file(self, filename, data):
        params = {'q': filename}
        response = await retry_transient_errors(
            self._session.post, self.objects_url, params=params, headers=self._headers, data=data
        )
        assert response.status == 200

    async def close(self):
        await self._session.close()
        self._session = None
        await self._fs.close()
        self._fs = None
