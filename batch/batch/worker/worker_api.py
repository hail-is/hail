import abc
from typing import Dict, Generic, List, TypedDict, TypeVar

from hailtop import httpx
from hailtop.aiocloud.common.credentials import CloudCredentials
from hailtop.aiotools.fs import AsyncFS
from hailtop.utils import CalledProcessError, sleep_before_try

from ..instance_config import InstanceConfig
from .disk import CloudDisk

CredsType = TypeVar("CredsType", bound=CloudCredentials)


class ContainerRegistryCredentials(TypedDict):
    username: str
    password: str


class CloudWorkerAPI(abc.ABC, Generic[CredsType]):
    nameserver_ip: str

    def __init__(self):
        self._user_credentials: Dict[str, CredsType] = {}
        self._jobs_per_user_credential: Dict[str, int] = {}

    @property
    @abc.abstractmethod
    def cloud_specific_env_vars_for_user_jobs(self) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def create_disk(self, instance_name: str, disk_name: str, size_in_gb: int, mount_path: str) -> CloudDisk:
        raise NotImplementedError

    @abc.abstractmethod
    def get_cloud_async_fs(self) -> AsyncFS:
        raise NotImplementedError

    @abc.abstractmethod
    def _load_user_credentials(self, credentials: Dict[str, str]) -> CredsType:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_user_hail_identity(self, credentials: Dict[str, str]) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    async def worker_container_registry_credentials(self, session: httpx.ClientSession) -> ContainerRegistryCredentials:
        raise NotImplementedError

    @abc.abstractmethod
    async def user_container_registry_credentials(self, hail_identity: str) -> ContainerRegistryCredentials:
        raise NotImplementedError

    @abc.abstractmethod
    def instance_config_from_config_dict(self, config_dict: Dict[str, str]) -> InstanceConfig:
        raise NotImplementedError

    @abc.abstractmethod
    async def _mount_cloudfuse(
        self,
        hail_identity: str,
        mount_base_path_data: str,
        mount_base_path_tmp: str,
        config: dict,
    ):
        raise NotImplementedError

    async def mount_cloudfuse(
        self,
        hail_identity: str,
        mount_base_path_data: str,
        mount_base_path_tmp: str,
        config: dict,
    ) -> None:
        tries = 0
        while True:
            try:
                return await self._mount_cloudfuse(hail_identity, mount_base_path_data, mount_base_path_tmp, config)
            except CalledProcessError:
                tries += 1
                if tries == 5:
                    raise

            await sleep_before_try(tries)

    @abc.abstractmethod
    async def unmount_cloudfuse(self, mount_base_path_data: str) -> None:
        raise NotImplementedError

    def register_user_credentials(self, credentials: Dict[str, str]) -> str:
        hail_identity = self._get_user_hail_identity(credentials)
        if hail_identity in self._user_credentials:
            assert hail_identity in self._jobs_per_user_credential
            self._jobs_per_user_credential[hail_identity] += 1
        else:
            self._user_credentials[hail_identity] = self._load_user_credentials(credentials)
            self._jobs_per_user_credential[hail_identity] = 1

        return hail_identity

    async def remove_user_credentials(self, hail_identity: str) -> None:
        rc = self._jobs_per_user_credential[hail_identity]
        if rc > 1:
            self._jobs_per_user_credential[hail_identity] -= 1
        else:
            del self._jobs_per_user_credential[hail_identity]
            await self._user_credentials.pop(hail_identity).close()

    @abc.abstractmethod
    async def close(self):
        raise NotImplementedError
