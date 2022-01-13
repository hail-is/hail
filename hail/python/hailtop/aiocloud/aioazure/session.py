from typing import Mapping, Optional, List, Union

from ..common import Session, AnonymousCloudCredentials
from .credentials import AzureCredentials


class AzureSession(Session):
    def __init__(self, *, credentials: Optional[Union[AzureCredentials, AnonymousCloudCredentials]] = None, credentials_file: Optional[str] = None,
                 params: Optional[Mapping[str, str]] = None, scopes: Optional[List[str]] = None, **kwargs):
        assert credentials is None or credentials_file is None, \
            f'specify only one of credentials or credentials_file: {(credentials, credentials_file)}'
        if credentials is None:
            if credentials_file:
                credentials = AzureCredentials.from_file(credentials_file, scopes=scopes)
            else:
                credentials = AzureCredentials.default_credentials(scopes=scopes)
        super().__init__(credentials=credentials, params=params, **kwargs)
