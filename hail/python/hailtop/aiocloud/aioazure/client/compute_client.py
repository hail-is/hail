from .base_client import AzureBaseClient


class AzureComputeClient(AzureBaseClient):
    def __init__(self, subscription_id, resource_group_name, **kwargs):
        if 'params' not in kwargs:
            kwargs['params'] = {}
        params = kwargs['params']
        if 'api-version' not in params:
            params['api-version'] = '2021-07-01'
        super().__init__(
            f'https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/providers/Microsoft.Compute',
            **kwargs,
        )
