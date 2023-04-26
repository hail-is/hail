import base64
import json
from typing import Dict

from ....worker.credentials import CloudUserCredentials


class AzureUserCredentials(CloudUserCredentials):
    def __init__(self, data: Dict[str, str]):
        self._data = data
        self._credentials = json.loads(base64.b64decode(data['key.json']).decode())

    @property
    def cloud_env_name(self) -> str:
        return 'AZURE_APPLICATION_CREDENTIALS'

    @property
    def username(self):
        return self._credentials['appId']

    @property
    def password(self):
        return self._credentials['password']

    @property
    def mount_path(self):
        return '/azure-credentials/key.json'

    def blobfuse_credentials(self, account: str, container: str) -> str:
        # https://github.com/Azure/azure-storage-fuse
        return f'''
accountName {account}
authType SPN
servicePrincipalClientId {self.username}
servicePrincipalClientSecret {self.password}
servicePrincipalTenantId {self._credentials['tenant']}
containerName {container}
'''
