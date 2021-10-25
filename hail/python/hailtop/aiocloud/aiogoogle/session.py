from typing import Mapping, Optional

from ..common import Session as BaseSession

from .credentials import GoogleCredentials


class GoogleSession(BaseSession):
    def __init__(self, *, credentials: Optional[GoogleCredentials] = None, params: Optional[Mapping[str, str]] = None, **kwargs):
        if credentials is None:
            credentials = GoogleCredentials.default_credentials()
        super().__init__(credentials=credentials, params=params, **kwargs)
