from typing import Mapping, Optional

from ..common import Session
from .credentials import AzureCredentials


class AzureSession(Session):
    def __init__(self, *, credentials: AzureCredentials = None, params: Optional[Mapping[str, str]] = None, **kwargs):
        if credentials is None:
            credentials = AzureCredentials.default_credentials()
        super().__init__(credentials=credentials, params=params, **kwargs)
