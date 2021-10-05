from typing import Optional

from ..credentials import AzureCredentials
from ..session import AzureSession
from .base_client import AzureBaseClient


class AzureResourcesManagementClient(AzureBaseClient):
    def __init__(self, subscription_id, resource_group_name, session: Optional[AzureSession] = None,
                 credentials: Optional[AzureCredentials] = None):
        session = session or AzureSession(credentials=credentials)
        super().__init__(f'https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/providers/Microsoft.Resources',
                         session=session)
