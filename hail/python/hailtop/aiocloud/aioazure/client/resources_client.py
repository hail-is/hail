from typing import Any, AsyncGenerator, Optional

from .base_client import AzureBaseClient


class AzureResourcesClient(AzureBaseClient):
    def __init__(self, subscription_id: str, **kwargs):
        super().__init__(f'https://management.azure.com/subscriptions/{subscription_id}', **kwargs)

    async def _list_resources(self, filter: Optional[str] = None) -> AsyncGenerator[Any, None]:
        # https://docs.microsoft.com/en-us/rest/api/resources/resources/list
        params = {'api-version': '2021-04-01'}
        if filter is not None:
            params['$filter'] = filter
        return self._paged_get('/resources', params=params)

    async def list_nic_names(self, machine_name_prefix: str) -> AsyncGenerator[str, None]:
        filter = f"resourceType eq 'Microsoft.Network/networkInterfaces' and substringof('{machine_name_prefix}',name)"
        async for resource in await self._list_resources(filter=filter):
            yield resource['name']

    async def list_public_ip_names(self, machine_name_prefix: str) -> AsyncGenerator[str, None]:
        filter = f"resourceType eq 'Microsoft.Network/publicIPAddresses' and substringof('{machine_name_prefix}',name)"
        async for resource in await self._list_resources(filter=filter):
            yield resource['name']
