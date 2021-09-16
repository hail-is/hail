import os
import json
import logging
from azure.identity.aio import DefaultAzureCredential, ClientSecretCredential

from hailtop.utils import first_extant_file

from ...common.auth import Credentials as BaseCredentials

log = logging.getLogger(__name__)


class Credentials(BaseCredentials):
    scopes = ['https://management.azure.com/']

    @staticmethod
    def from_file(credentials_file):
        with open(credentials_file, 'r') as f:
            data = json.loads(f.read())
            credential = ClientSecretCredential(tenant_id=data['tenant'],
                                                client_id=data['appId'],
                                                client_secret=data['password'])
            return Credentials(credential)

    @staticmethod
    def default_credentials():
        credentials_file = first_extant_file(
            os.environ.get('AZURE_APPLICATION_CREDENTIALS'),
            '/azure-key/credentials.json'
        )

        if credentials_file:
            log.info(f'using credentials file {credentials_file}')
            return Credentials.from_file(credentials_file)

        return Credentials(DefaultAzureCredential())

    def __init__(self, credential):
        self.credential = credential

    async def get_access_token(self, session):
        return await self.credential.get_token(*Credentials.scopes)

    async def close(self):
        await self.credential.close()
