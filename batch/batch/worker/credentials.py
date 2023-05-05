import abc


class CloudUserCredentials(abc.ABC):
    @property
    @abc.abstractmethod
    def cloud_env_name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def username(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def password(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mount_path(self):
        raise NotImplementedError

    @abc.abstractmethod
    def cloudfuse_credentials(self, fuse_config: dict) -> str:
        raise NotImplementedError
