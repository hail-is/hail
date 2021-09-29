from typing import Mapping, Optional

from ..common import Session as BaseSession

from .auth import AccessToken, Credentials


class GCPSession(BaseSession):
    def __init__(self, *, credentials: Credentials = None, params: Optional[Mapping[str, str]] = None, **kwargs):
        if credentials is None:
            credentials = Credentials.default_credentials()
        access_token = AccessToken(credentials)
        super().__init__(access_token=access_token, params=params, **kwargs)
