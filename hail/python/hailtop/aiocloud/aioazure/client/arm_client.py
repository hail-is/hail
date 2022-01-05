from typing import AsyncGenerator, Any, Optional

from .base_client import AzureBaseClient


class AzureResourceManagerClient(AzureBaseClient):
    def __init__(self, subscription_id: str, resource_group_name: str, **kwargs):
        if 'params' not in kwargs:
            kwargs['params'] = {}
        params = kwargs['params']
        if 'api-version' not in params:
            params['api-version'] = '2021-04-01'
        super().__init__(f'https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/providers/Microsoft.Resources',
                         **kwargs)

    async def list_deployments(self, filter: Optional[str] = None) -> AsyncGenerator[Any, None]:
        # https://docs.microsoft.com/en-us/rest/api/resources/deployments/list-by-resource-group
        params = {}
        if filter is not None:
            params['$filter'] = filter
        return self._paged_get('/deployments', params=params)
