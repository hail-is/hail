from typing import Dict
import base64

from ....worker.credentials import CloudUserCredentials


class GCPUserCredentials(CloudUserCredentials):
    def __init__(self, data: Dict[str, bytes]):
        self.secret_data = data
        self.secret_name = 'gsa-key'
        self.file_name = 'key.json'
        self.cloud_env_name = 'GOOGLE_APPLICATION_CREDENTIALS'
        self.hail_env_name = 'HAIL_GSA_KEY_FILE'
        self.username = '_json_key'

    @property
    def password(self) -> str:
        return base64.b64decode(self.secret_data['key.json']).decode()
