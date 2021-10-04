import os
import json
import time
import logging
from azure.identity.aio import DefaultAzureCredential, ClientSecretCredential

from hailtop.utils import first_extant_file

from ..common.credentials import CloudCredentials

log = logging.getLogger(__name__)


class AzureCredentials(CloudCredentials):
    scopes = ['https://management.azure.com/']

    @staticmethod
    def from_file(credentials_file):
        with open(credentials_file, 'r') as f:
            data = json.loads(f.read())
            credential = ClientSecretCredential(tenant_id=data['tenant'],
                                                client_id=data['appId'],
                                                client_secret=data['password'])
            return AzureCredentials(credential)

    @staticmethod
    def default_credentials():
        credentials_file = first_extant_file(
            os.environ.get('AZURE_APPLICATION_CREDENTIALS'),
            '/azure-key/credentials.json'
        )

        if credentials_file:
            log.info(f'using credentials file {credentials_file}')
            return AzureCredentials.from_file(credentials_file)

        return AzureCredentials(DefaultAzureCredential())

    def __init__(self, credential):
        self.credential = credential
        self._access_token = None
        self._expires_at = None

    async def auth_headers(self):
        now = time.time()
        if self._access_token is None or now > self._expires_at:
            self._access_token = await self.get_access_token()
            self._expires_at = now + (self._access_token.expires_on - now) // 2
        return {'Authorization': f'Bearer {self._access_token.token}'}

    async def get_access_token(self):
        return await self.credential.get_token(*AzureCredentials.scopes)

    async def close(self):
        await self.credential.close()
