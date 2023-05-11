import abc


class CloudUserCredentials(abc.ABC):
    @property
    @abc.abstractmethod
    def cloud_env_name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mount_path(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def identity_json(self) -> dict:
        raise NotImplementedError
