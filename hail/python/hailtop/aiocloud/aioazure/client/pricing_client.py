from typing import Optional, AsyncGenerator, Any

from .base_client import AzureBaseClient


class AzurePricingClient(AzureBaseClient):
    def __init__(self, **kwargs):
        super().__init__('https://prices.azure.com/api/retail', **kwargs)

    async def _paged_get(self, path, **kwargs) -> AsyncGenerator[Any, None]:
        page = await self.get(path, **kwargs)
        for v in page.get('Items', []):
            yield v
        next_link = page.get('NextPageLink')
        while next_link is not None:
            page = await self.get(url=next_link)
            for v in page['Items']:
                yield v
            next_link = page.get('NextPageLink')

    def list_prices(self, filter: Optional[str] = None) -> AsyncGenerator[Any, None]:
        # https://docs.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices
        params = {}
        if filter is not None:
            params['$filter'] = filter
        return self._paged_get('/prices', params=params)
