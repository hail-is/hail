from typing import Any, AsyncGenerator

from .base_client import GoogleBaseClient


class GoogleBillingClient(GoogleBaseClient):
    def __init__(self, **kwargs):
        super().__init__('https://cloudbilling.googleapis.com/v1/services', **kwargs)

    # https://cloud.google.com/billing/docs/reference/rest/v1/services.skus/list
    async def list_skus(self, path, **kwargs) -> AsyncGenerator[Any, None]:
        params = kwargs.get('params', {})
        page = await self.get(path, **kwargs)
        for v in page.get('skus', []):
            yield v
        next_page_token = page.get('nextPageToken')
        while next_page_token:  # This field is empty if there are no more results to retrieve.
            params['pageToken'] = next_page_token
            page = await self.get(path, params=params)
            for v in page['skus']:
                yield v
            next_page_token = page.get('nextPageToken')
