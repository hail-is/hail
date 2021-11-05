from typing import Dict
import base64
import json

from ....worker.credentials import CloudUserCredentials


class AzureUserCredentials(CloudUserCredentials):
    def __init__(self, data: Dict[str, bytes]):
        self.secret_data = data
        self.secret_name = 'azure-credentials'
        self.file_name = 'key.json'
        self.cloud_env_name = 'AZURE_APPLICATION_CREDENTIALS'
        self.hail_env_name = 'HAIL_AZURE_CREDENTIAL_FILE'
        self.credentials = json.loads(base64.b64decode(self.secret_data['key.json']).decode())

    @property
    def username(self):
        return self.credentials['appId']

    @property
    def password(self):
        return self.credentials['password']
