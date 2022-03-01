import base64
from typing import Dict

from ....worker.credentials import CloudUserCredentials


class GCPUserCredentials(CloudUserCredentials):
    def __init__(self, data: Dict[str, bytes]):
        self._data = data

    @property
    def secret_name(self) -> str:
        return 'gsa-key'

    @property
    def secret_data(self) -> Dict[str, bytes]:
        return self._data

    @property
    def file_name(self) -> str:
        return 'key.json'

    @property
    def cloud_env_name(self) -> str:
        return 'GOOGLE_APPLICATION_CREDENTIALS'

    @property
    def hail_env_name(self) -> str:
        return 'HAIL_GSA_KEY_FILE'

    @property
    def username(self):
        return '_json_key'

    @property
    def password(self) -> str:
        return base64.b64decode(self.secret_data['key.json']).decode()

    @property
    def mount_path(self):
        return f'/{self.secret_name}/{self.file_name}'

    def cloudfuse_credentials(self, fuse_config: dict) -> str:  # pylint: disable=unused-argument
        return self.password
