import abc
from typing import Dict


class CloudUserCredentials(abc.ABC):
    secret_name: str
    secret_data: Dict[str, bytes]
    file_name: str
    cloud_env_name: str
    hail_env_name: str
    username: str

    @property
    def password(self) -> str:
        raise NotImplementedError

    @property
    def mount_path(self):
        return f'/{self.secret_name}/{self.file_name}'
