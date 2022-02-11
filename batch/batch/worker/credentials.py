import abc
from typing import Dict


class CloudUserCredentials(abc.ABC):
    @property
    def secret_name(self) -> str:
        raise NotImplementedError

    @property
    def secret_data(self) -> Dict[str, bytes]:
        raise NotImplementedError

    @property
    def file_name(self) -> str:
        raise NotImplementedError

    @property
    def cloud_env_name(self) -> str:
        raise NotImplementedError

    @property
    def hail_env_name(self) -> str:
        raise NotImplementedError

    @property
    def username(self) -> str:
        raise NotImplementedError

    @property
    def password(self) -> str:
        raise NotImplementedError

    @property
    def mount_path(self):
        return f'/{self.secret_name}/{self.file_name}'

    @abc.abstractmethod
    def cloudfuse_credentials(self, fuse_config: dict) -> str:
        raise NotImplementedError
