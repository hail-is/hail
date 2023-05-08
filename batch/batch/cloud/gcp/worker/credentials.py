import base64
from typing import Dict

from ....worker.credentials import CloudUserCredentials


class GCPUserCredentials(CloudUserCredentials):
    def __init__(self, data: Dict[str, str]):
        self._data = data

    @property
    def cloud_env_name(self) -> str:
        return 'GOOGLE_APPLICATION_CREDENTIALS'

    @property
    def username(self):
        return '_json_key'

    @property
    def password(self) -> str:
        return base64.b64decode(self._data['key.json']).decode()

    @property
    def mount_path(self):
        return '/gsa-key/key.json'

    def cloudfuse_credentials(self, fuse_config: dict) -> str:  # pylint: disable=unused-argument
        return self.password
