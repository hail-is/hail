from typing import Mapping, Optional

from ..common import Session

from .auth import AccessToken, Credentials


class AzureSession(Session):
    def __init__(self, *, credentials: Credentials = None, params: Optional[Mapping[str, str]] = None, **kwargs):
        if credentials is None:
            credentials = Credentials.default_credentials()
        self.credentials = credentials
        access_token = AccessToken(credentials)
        super().__init__(access_token=access_token, params=params, **kwargs)

    async def close(self):
        try:
            await super().close()
        finally:
            await self.credentials.close()
