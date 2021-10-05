from typing import Optional, Mapping, Any

from ..credentials import AzureCredentials
from ..session import AzureSession
from .base_client import AzureBaseClient


class AzurePagedEntryIterator:
    def __init__(self, client: 'AzureResourcesClient', path: str, params: Mapping[str, Any]):
        self._client = client
        self._path = path
        self._params = params
        self._page = None
        self._entry_index = None

    def __aiter__(self) -> 'AzurePagedEntryIterator':
        return self

    async def __anext__(self):
        if self._page is None:
            self._page = await self._client.get(
                self._path, params=self._params)
            self._entry_index = 0

        # in case a response is empty but there are more pages
        while True:
            # an empty page has no entries
            if 'value' in self._page and self._entry_index < len(self._page['value']):
                i = self._entry_index
                self._entry_index += 1
                return self._page['value'][i]

            next_link = self._page.get('nextLink')
            if next_link is not None:
                self._page = await self._client.get_next_link(next_link)
                self._entry_index = 0
            else:
                raise StopAsyncIteration


class AzureResourcesClient(AzureBaseClient):
    def __init__(self, subscription_id, resource_group_name, session: Optional[AzureSession] = None,
                 credentials: Optional[AzureCredentials] = None):
        session = session or AzureSession(credentials=credentials)
        super().__init__(f'https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}',
                         session=session)

    async def list_resources(self, filter: Optional[str] = None):
        params = {
            'api-version': '2021-04-01'
        }

        if filter:
            params['$filter'] = filter

        return AzurePagedEntryIterator(self, '/resources', params=params)
