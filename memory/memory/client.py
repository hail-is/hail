from asyncinit import asyncinit
import aiohttp
import concurrent

from hailtop.auth import service_auth_headers
from hailtop.config import get_deploy_config
from hailtop.google_storage import GCS
from hailtop.tls import get_context_specific_ssl_client_session
from hailtop.utils import async_to_blocking, request_retry_transient_errors

@asyncinit
class AsyncMemoryClient:
    async def __init__(self, billing_project, fs=None, deploy_config=None, session=None,
                       headers=None, _token=None):
        self.billing_project = billing_project

        if not deploy_config:
            deploy_config = get_deploy_config()

        self.url = deploy_config.base_url('memory')

        if session is None:
            session = get_context_specific_ssl_client_session(
                raise_for_status=True,
                timeout=aiohttp.ClientTimeout(total=60))
        self._session = session

        if fs is None:
            fs = GCS(blocking_pool=concurrent.futures.ThreadPoolExecutor(), project=billing_project)
        self._fs = fs

        h = {}
        if headers:
            h.update(headers)
        if _token:
            h['Authorization'] = f'Bearer {_token}'
        else:
            h.update(service_auth_headers(deploy_config, 'memory'))
        self._headers = h

    async def _get_file_if_exists(self, filename):
        request = {
            'billing_project': self.billing_project,
            'filename': filename}
        try:
            response = await request_retry_transient_errors(
                self._session, 'POST',
                self.url + '/getfile', json=request, headers=self._headers)
            async with response:
                if response.status == 200:
                    return await response.read()
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                return None
            raise e

    async def read_file(self, filename):
        data = await self._get_file_if_exists(filename)
        if data is not None:
            return data
        return await self._fs.read_binary_gs_file(filename)

    async def close(self):
        await self._session.close()
        self._session = None

class BlockingMemoryClient:
    def __init__(self, billing_project, fs=None, deploy_config=None, session=None,
                 headers=None, _token=None):
        self._client = async_to_blocking(AsyncMemoryClient(billing_project, fs, deploy_config, session, headers, _token))

    def _get_file_if_exists(self, filename):
        return async_to_blocking(self._client._get_file_if_exists(filename))

    def read_file(self, filename):
        return async_to_blocking(self._client.read_file(filename))

    def close(self):
        return async_to_blocking(self._client.close())
