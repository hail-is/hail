from typing import Optional, ClassVar, List

from ..session import AzureSession
from .base_client import AzureBaseClient


class AzureGraphClient(AzureBaseClient):
    required_scopes: ClassVar[List[str]] = ['https://graph.microsoft.com/.default']

    def __init__(self, session: Optional[AzureSession] = None, **kwargs):
        if 'scopes' in kwargs:
            kwargs['scopes'] += AzureGraphClient.required_scopes
        else:
            kwargs['scopes'] = AzureGraphClient.required_scopes
        session = session or AzureSession(**kwargs)
        super().__init__('https://graph.microsoft.com/v1.0', session=session)
