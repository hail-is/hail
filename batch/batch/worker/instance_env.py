import abc
from typing import Dict

from hailtop import httpx

from .disk import CloudDisk
from .credentials import CloudUserCredentials
from ..instance_config import InstanceConfig


class CloudWorkerAPI(abc.ABC):
    @property
    @abc.abstractmethod
    def nameserver_ip(self):
        raise NotImplementedError

    @abc.abstractmethod
    def create_disk(self, instance_name: str, disk_name: str, size_in_gb: int, mount_path: str) -> CloudDisk:
        raise NotImplementedError

    @abc.abstractmethod
    def user_credentials(self, credentials: Dict[str, bytes]) -> CloudUserCredentials:
        raise NotImplementedError

    @abc.abstractmethod
    async def worker_access_token(self, session: httpx.ClientSession) -> Dict[str, str]:
        raise NotImplementedError

    @abc.abstractmethod
    def instance_config_from_config_dict(self, config_dict: Dict[str, str]) -> InstanceConfig:
        raise NotImplementedError
