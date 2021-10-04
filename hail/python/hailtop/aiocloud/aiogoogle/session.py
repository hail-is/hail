from typing import Mapping, Optional

from ..common import Session as BaseSession

from .credentials import GCPCredentials


class GCPSession(BaseSession):
    def __init__(self, *, credentials: Optional[GCPCredentials] = None, params: Optional[Mapping[str, str]] = None, **kwargs):
        if credentials is None:
            credentials = GCPCredentials.default_credentials()
        super().__init__(credentials=credentials, params=params, **kwargs)
