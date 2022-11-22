from typing import Any, Mapping, MutableMapping, Optional
from .base_client import GoogleBaseClient


class PagedEntryIterator:
    def __init__(self, client: 'GoogleLoggingClient', body: MutableMapping[str, Any], request_kwargs: Mapping[str, Any]):
        self._client = client
        self._body = body
        self._request_kwargs = request_kwargs
        self._page = None
        self._entry_index: Optional[int] = None

    def __aiter__(self) -> 'PagedEntryIterator':
        return self

    async def __anext__(self):
        if self._page is None:
            assert 'pageToken' not in self._body
            self._page = await self._client.post(
                '/entries:list', json=self._body, **self._request_kwargs)
            self._entry_index = 0

        # in case a response is empty but there are more pages
        while True:
            assert self._page
            # an empty page has no entries
            if 'entries' in self._page and self._entry_index is not None and self._entry_index < len(self._page['entries']):
                i = self._entry_index
                self._entry_index += 1
                return self._page['entries'][i]

            next_page_token = self._page.get('nextPageToken')
            if next_page_token is not None:
                self._body['pageToken'] = next_page_token
                self._page = await self._client.post(
                    '/entries:list', json=self._body, **self._request_kwargs)
                self._entry_index = 0
            else:
                raise StopAsyncIteration


class GoogleLoggingClient(GoogleBaseClient):
    def __init__(self, **kwargs):
        super().__init__('https://logging.googleapis.com/v2', **kwargs)

    # docs:
    # https://cloud.google.com/logging/docs/reference/v2/rest

    # https://cloud.google.com/logging/docs/reference/v2/rest/v2/entries/list
    async def list_entries(self, *, body: MutableMapping[str, Any], **kwargs):
        return PagedEntryIterator(self, body, kwargs)
