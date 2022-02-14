import abc
from typing import Dict

from hailtop import httpx
from hailtop.utils import check_shell, CalledProcessError, sleep_and_backoff

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

    @abc.abstractmethod
    def write_cloudfuse_credentials(self, root_dir: str, credentials: str, bucket: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def _mount_cloudfuse(
        self, fuse_credentials_path: str, mount_base_path_data: str, mount_base_path_tmp: str, config: dict
    ) -> str:
        raise NotImplementedError

    async def mount_cloudfuse(
        self, credentials_path: str, mount_base_path_data: str, mount_base_path_tmp: str, config: dict
    ) -> None:
        mount_command = self._mount_cloudfuse(credentials_path, mount_base_path_data, mount_base_path_tmp, config)
        delay = 0.1
        error = 0
        while True:
            try:
                return await check_shell(mount_command)
            except CalledProcessError:
                error += 1
                if error == 5:
                    raise

            delay = await sleep_and_backoff(delay)

    @abc.abstractmethod
    def _unmount_cloudfuse(self, mount_base_path: str) -> str:
        raise NotImplementedError

    async def unmount_cloudfuse(self, mount_base_path: str) -> None:
        await check_shell(self._unmount_cloudfuse(mount_base_path))
