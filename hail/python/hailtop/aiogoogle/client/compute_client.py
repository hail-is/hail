from typing import Mapping, Any
from .base_client import BaseClient


class PagedIterator:
    def __init__(self, client: 'ComputeClient', path: str, request_params: Mapping[str, Any], request_kwargs: Mapping[str, Any]):
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

    async def list(self, path: str, *, params: Mapping[str, Any] = None, **kwargs) -> Any:
        return PagedIterator(self, path, params, kwargs)
