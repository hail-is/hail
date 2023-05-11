import abc

from hailtop.auth.auth import IdentityProvider


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
    def identity_provider(self) -> IdentityProvider:
        raise NotImplementedError
