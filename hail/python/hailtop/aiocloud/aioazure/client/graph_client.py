from typing import ClassVar, List

from .base_client import AzureBaseClient


class AzureGraphClient(AzureBaseClient):
    required_scopes: ClassVar[List[str]] = ['https://graph.microsoft.com/.default']

    def __init__(self, **kwargs):
        if 'scopes' in kwargs:
            kwargs['scopes'] += AzureGraphClient.required_scopes
        else:
            kwargs['scopes'] = AzureGraphClient.required_scopes
        super().__init__('https://graph.microsoft.com/v1.0', **kwargs)
