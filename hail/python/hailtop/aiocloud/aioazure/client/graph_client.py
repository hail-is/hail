from typing import Optional

from ..session import AzureSession
from .base_client import AzureBaseClient


class AzureGraphClient(AzureBaseClient):
    def __init__(self, session: Optional[AzureSession] = None, **kwargs):
        session = session or AzureSession(**kwargs)
        super().__init__('https://graph.microsoft.com/v1.0', session=session)
