import concurrent.futures
import os
import json
import time
import logging

from typing import Any, List, Optional, Union
from azure.identity.aio import DefaultAzureCredential, ClientSecretCredential
from azure.core.credentials import AccessToken
from azure.core.credentials_async import AsyncTokenCredential

import msal

from hailtop.utils import first_extant_file, blocking_to_async

from ..common.credentials import CloudCredentials

log = logging.getLogger(__name__)


class RefreshTokenCredential(AsyncTokenCredential):
    def __init__(self, client_id: str, tenant_id: str, refresh_token: str):
        authority = f'https://login.microsoftonline.com/{tenant_id}'
        self._app = msal.PublicClientApplication(client_id, authority=authority)
        self._pool = concurrent.futures.ThreadPoolExecutor()
        self._refresh_token: Optional[str] = refresh_token

    async def get_token(
        self, *scopes: str, claims: Optional[str] = None, tenant_id: Optional[str] = None, **kwargs: Any
    ) -> AccessToken:
        # MSAL token objects, like those returned from `acquire_token_by_refresh_token` do their own internal
        # caching of refresh tokens. Per their documentation it is not advised to use the original refresh token
        # once you have "migrated it into MSAL".
        # See docs:
        # https://msal-python.readthedocs.io/en/latest/#msal.ClientApplication.acquire_token_by_refresh_token
        if self._refresh_token:
            res_co = blocking_to_async(self._pool, self._app.acquire_token_by_refresh_token, self._refresh_token, scopes)
            self._refresh_token = None
            res = await res_co
        else:
            res = await blocking_to_async(self._pool, self._app.acquire_token_silent, scopes, None)
            assert res
        return AccessToken(res['access_token'], res['id_token_claims']['exp'])

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

    async def close(self) -> None:
        self._pool.shutdown()


class AzureCredentials(CloudCredentials):
    @staticmethod
    def from_credentials_data(credentials: dict, scopes: Optional[List[str]] = None):
        if 'refreshToken' in credentials:
            return AzureCredentials(
                RefreshTokenCredential(
                    client_id=credentials['appId'],
                    tenant_id=credentials['tenant'],
                    refresh_token=credentials['refreshToken'],
                ),
                scopes=scopes,
            )

        assert 'password' in credentials
        return AzureCredentials(
            ClientSecretCredential(
                tenant_id=credentials['tenant'],
                client_id=credentials['appId'],
                client_secret=credentials['password']
            ),
            scopes
        )

    @staticmethod
    def from_file(credentials_file: str, scopes: Optional[List[str]] = None):
        with open(credentials_file, 'r', encoding='utf-8') as f:
            credentials = json.loads(f.read())
            return AzureCredentials.from_credentials_data(credentials, scopes)

    @staticmethod
    def default_credentials(scopes: Optional[List[str]] = None):
        credentials_file = first_extant_file(
            os.environ.get('AZURE_APPLICATION_CREDENTIALS'),
            '/azure-credentials/credentials.json',
            '/gsa-key/key.json'  # FIXME: make this file path cloud-agnostic
        )

        if credentials_file:
            log.info(f'using credentials file {credentials_file}')
            return AzureCredentials.from_file(credentials_file, scopes)

        return AzureCredentials(DefaultAzureCredential(), scopes)

    def __init__(self, credential: Union[DefaultAzureCredential, ClientSecretCredential, RefreshTokenCredential], scopes: Optional[List[str]] = None):
        self.credential = credential
        self._access_token = None
        self._expires_at = None

        if scopes is None:
            scopes = ['https://management.azure.com/.default']
        self.scopes = scopes

    async def auth_headers(self):
        access_token = await self.access_token()
        return {'Authorization': f'Bearer {access_token}'}

    async def access_token(self) -> str:
        now = time.time()
        if self._access_token is None or (self._expires_at is not None and now > self._expires_at):
            self._access_token = await self.get_access_token()
            self._expires_at = now + (self._access_token.expires_on - now) // 2   # type: ignore
        assert self._access_token
        return self._access_token.token

    async def get_access_token(self):
        return await self.credential.get_token(*self.scopes)

    async def close(self):
        await self.credential.close()
