import abc
from typing import Dict, Generic, TypeVar

from hailtop import httpx
from hailtop.aiotools.fs import AsyncFS
from hailtop.utils import CalledProcessError, sleep_and_backoff

from ..instance_config import InstanceConfig
from .credentials import CloudUserCredentials
from .disk import CloudDisk

CredsType = TypeVar("CredsType", bound=CloudUserCredentials)


class CloudWorkerAPI(abc.ABC, Generic[CredsType]):
    nameserver_ip: str

    @abc.abstractmethod
    def get_compute_client(self):
        raise NotImplementedError

    @abc.abstractmethod
    def create_disk(self, instance_name: str, disk_name: str, size_in_gb: int, mount_path: str) -> CloudDisk:
        raise NotImplementedError

    @abc.abstractmethod
    def get_cloud_async_fs(self) -> AsyncFS:
        raise NotImplementedError

    @abc.abstractmethod
    def user_credentials(self, credentials: Dict[str, str]) -> CredsType:
        raise NotImplementedError

    @abc.abstractmethod
    async def worker_access_token(self, session: httpx.ClientSession) -> Dict[str, str]:
        raise NotImplementedError

    @abc.abstractmethod
    def instance_config_from_config_dict(self, config_dict: Dict[str, str]) -> InstanceConfig:
        raise NotImplementedError

    @abc.abstractmethod
    async def _mount_cloudfuse(
        self,
        credentials: CredsType,
        mount_base_path_data: str,
        mount_base_path_tmp: str,
        config: dict,
    ):
        raise NotImplementedError

    async def mount_cloudfuse(
        self,
        credentials: CredsType,
        mount_base_path_data: str,
        mount_base_path_tmp: str,
        config: dict,
    ) -> None:
        delay = 0.1
        error = 0
        while True:
            try:
                return await self._mount_cloudfuse(credentials, mount_base_path_data, mount_base_path_tmp, config)
            except CalledProcessError:
                error += 1
                if error == 5:
                    raise

            delay = await sleep_and_backoff(delay)

    @abc.abstractmethod
    async def unmount_cloudfuse(self, mount_base_path_data: str) -> None:
        raise NotImplementedError
