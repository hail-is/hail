import os
import json
import time
import logging
from typing import List, Optional
from azure.identity.aio import DefaultAzureCredential, ClientSecretCredential

from hailtop.utils import first_extant_file

from ..common.credentials import CloudCredentials

log = logging.getLogger(__name__)


class AzureCredentials(CloudCredentials):
    @staticmethod
    def from_credentials_data(credentials: dict, scopes: Optional[List[str]] = None):
        credential = ClientSecretCredential(tenant_id=credentials['tenant'],
                                            client_id=credentials['appId'],
                                            client_secret=credentials['password'])
        return AzureCredentials(credential, scopes)

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

    def __init__(self, credential, scopes: Optional[List[str]] = None):
        self.credential = credential
        self._access_token = None
        self._expires_at = None

        if scopes is None:
            scopes = ['https://management.azure.com/.default']
        self.scopes = scopes

    async def auth_headers(self):
        now = time.time()
        if self._access_token is None or (self._expires_at is not None and now > self._expires_at):
            self._access_token = await self.get_access_token()
            self._expires_at = now + (self._access_token.expires_on - now) // 2   # type: ignore
        return {'Authorization': f'Bearer {self._access_token.token}'}  # type: ignore

    async def get_access_token(self):
        return await self.credential.get_token(*self.scopes)

    async def close(self):
        await self.credential.close()
