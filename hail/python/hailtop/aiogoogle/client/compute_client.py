import uuid
from typing import Mapping, Any, Optional, MutableMapping

from .base_client import BaseClient
from hailtop.utils import sleep_and_backoff


async def request_with_wait_for_done(request_f, path, params: MutableMapping[str, Any] = None, **kwargs):
    assert 'params' not in kwargs

    if params is None:
        params = {}

    request_uuid = str(uuid.uuid4())
    if 'requestId' not in params:
        params['requestId'] = request_uuid

    delay = 0.2
    while True:
        resp = await request_f(path, params=params, **kwargs)
        if resp['status'] == 'DONE':
            return resp
        delay = await sleep_and_backoff(delay)


class PagedIterator:
    def __init__(self, client: 'ComputeClient', path: str, request_params: Optional[Mapping[str, Any]], request_kwargs: Mapping[str, Any]):
        assert 'params' not in request_kwargs
        self._client = client
        self._path = path
        if request_params is None:
            request_params = {}
        self._request_params = request_params
        self._request_kwargs = request_kwargs
        self._page = None
        self._index = None

    def __aiter__(self) -> 'PagedIterator':
        return self

    async def __anext__(self):
        if self._page is None:
            assert 'pageToken' not in self._request_params
            self._page = await self._client.get(self._path, params=self._request_params, **self._request_kwargs)
            self._index = 0

        while True:
            if 'items' in self._page and self._index < len(self._page['items']):
                i = self._index
                self._index += 1
                return self._page['items'][i]

            next_page_token = self._page.get('nextPageToken')
            if next_page_token is not None:
                self._request_params['pageToken'] = next_page_token
                self._page = await self._client.get(self._path, params=self._request_params, **self._request_kwargs)
                self._index = 0
            else:
                raise StopAsyncIteration


class ComputeClient(BaseClient):
    def __init__(self, project, **kwargs):
        super().__init__(f'https://compute.googleapis.com/compute/v1/projects/{project}', **kwargs)

    # docs:
    # https://cloud.google.com/compute/docs/reference/rest/v1
    # https://cloud.google.com/compute/docs/reference/rest/v1/instances/insert
    # https://cloud.google.com/compute/docs/reference/rest/v1/instances/get
    # https://cloud.google.com/compute/docs/reference/rest/v1/instances/delete
    # https://cloud.google.com/compute/docs/reference/rest/v1/disks

    async def list(self, path: str, *, params: MutableMapping[str, Any] = None, **kwargs) -> PagedIterator:
        return PagedIterator(self, path, params, kwargs)

    async def create_disk(self, path: str, *, params: MutableMapping[str, Any] = None, **kwargs):
        return await request_with_wait_for_done(self.post, path, params, **kwargs)

    async def attach_disk(self, path: str, *, params: MutableMapping[str, Any] = None, **kwargs):
        return await request_with_wait_for_done(self.post, path, params, **kwargs)

    async def detach_disk(self, path: str, *, params: MutableMapping[str, Any] = None, **kwargs):
        return await request_with_wait_for_done(self.post, path, params, **kwargs)

    async def delete_disk(self, path: str, *, params: MutableMapping[str, Any] = None, **kwargs):
        return await request_with_wait_for_done(self.delete, path, params, **kwargs)
