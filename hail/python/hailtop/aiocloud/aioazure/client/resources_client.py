from typing import Optional

from ..session import AzureSession
from .base_client import AzureBaseClient


class AzureResourcesClient(AzureBaseClient):
    def __init__(self, subscription_id, resource_group_name, session: Optional[AzureSession] = None):
        session = session or AzureSession()
        super().__init__(f'https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/providers/Microsoft.Resources',
                         session=session)
