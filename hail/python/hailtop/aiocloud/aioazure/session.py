from typing import Mapping, Optional, List, Union

from ..common import Session
from .credentials import AzureCredentials, EmptyAzureCredentials


class AzureSession(Session):
    def __init__(self, *, credentials: Union[AzureCredentials, EmptyAzureCredentials] = None, credentials_file: Optional[str] = None,
                 params: Optional[Mapping[str, str]] = None, scopes: Optional[List[str]] = None, **kwargs):
        if credentials is None:
            if credentials_file:
                credentials = AzureCredentials.from_file(credentials_file, scopes=scopes)
            else:
                credentials = AzureCredentials.default_credentials(scopes=scopes)
        super().__init__(credentials=credentials, params=params, **kwargs)
