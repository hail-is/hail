from typing import Mapping, Optional

from ..common import Session as BaseSession

from .credentials import GoogleCredentials


class GoogleSession(BaseSession):
    def __init__(self, *, credentials: Optional[GoogleCredentials] = None, credentials_file: Optional[str] = None,
                 params: Optional[Mapping[str, str]] = None, **kwargs):
        if credentials is None:
            if credentials_file:
                credentials = GoogleCredentials.from_file(credentials_file)
            else:
                credentials = GoogleCredentials.default_credentials()
        super().__init__(credentials=credentials, params=params, **kwargs)
