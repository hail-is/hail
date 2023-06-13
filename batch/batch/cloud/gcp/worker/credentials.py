import base64
from typing import Dict

from ....worker.credentials import CloudUserCredentials


class GCPUserCredentials(CloudUserCredentials):
    def __init__(self, data: Dict[str, str]):
        self._data = data
        self._key = base64.b64decode(self._data['key.json']).decode()

    @property
    def cloud_env_name(self) -> str:
        return 'GOOGLE_APPLICATION_CREDENTIALS'

    @property
    def mount_path(self):
        return '/gsa-key/key.json'

    @property
    def key(self):
        return self._key
