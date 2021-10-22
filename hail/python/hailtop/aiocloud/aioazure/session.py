from typing import Mapping, Optional, List

from ..common import Session
from .credentials import AzureCredentials


class AzureSession(Session):
    def __init__(self, *, credentials: AzureCredentials = None, credentials_file: Optional[str] = None,
                 params: Optional[Mapping[str, str]] = None, scopes: Optional[List[str]] = None, **kwargs):
        if credentials is None:
            if credentials_file:
                credentials = AzureCredentials.from_file(credentials_file, scopes=scopes)
            else:
                credentials = AzureCredentials.default_credentials(scopes=scopes)
        super().__init__(credentials=credentials, params=params, **kwargs)
