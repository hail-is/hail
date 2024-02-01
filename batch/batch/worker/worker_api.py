import abc
from typing import Dict, List, TypedDict

from aiohttp import web

from hailtop import httpx
from hailtop.utils import CalledProcessError, sleep_before_try

from ..instance_config import InstanceConfig
from .disk import CloudDisk


class ContainerRegistryCredentials(TypedDict):
    username: str
    password: str


class CloudWorkerAPI(abc.ABC):
    nameserver_ip: str

    @property
    @abc.abstractmethod
    def cloud_specific_env_vars_for_user_jobs(self) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def create_disk(self, instance_name: str, disk_name: str, size_in_gb: int, mount_path: str) -> CloudDisk:
        raise NotImplementedError

    @abc.abstractmethod
    async def worker_container_registry_credentials(self, session: httpx.ClientSession) -> ContainerRegistryCredentials:
        raise NotImplementedError

    @abc.abstractmethod
    async def user_container_registry_credentials(self, credentials: Dict[str, str]) -> ContainerRegistryCredentials:
        raise NotImplementedError

    @abc.abstractmethod
    def create_metadata_server_app(self, credentials: Dict[str, str]) -> web.Application:
        raise NotImplementedError

    @abc.abstractmethod
    def instance_config_from_config_dict(self, config_dict: Dict[str, str]) -> InstanceConfig:
        raise NotImplementedError

    @abc.abstractmethod
    async def _mount_cloudfuse(
        self,
        credentials: Dict[str, str],
        mount_base_path_data: str,
        mount_base_path_tmp: str,
        config: dict,
    ):
        raise NotImplementedError

    async def mount_cloudfuse(
        self,
        credentials: Dict[str, str],
        mount_base_path_data: str,
        mount_base_path_tmp: str,
        config: dict,
    ) -> None:
        tries = 0
        while True:
            try:
                return await self._mount_cloudfuse(credentials, mount_base_path_data, mount_base_path_tmp, config)
            except CalledProcessError:
                tries += 1
                if tries == 5:
                    raise

            await sleep_before_try(tries)

    @abc.abstractmethod
    async def unmount_cloudfuse(self, mount_base_path_data: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self):
        raise NotImplementedError
