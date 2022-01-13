import aiohttp
from typing import Dict
import os

from hailtop.utils import request_retry_transient_errors
from hailtop import httpx

from ....worker.instance_env import CloudWorkerAPI
from .disk import GCPDisk
from .credentials import GCPUserCredentials
from ..instance_config import GCPSlimInstanceConfig


class GCPWorkerAPI(CloudWorkerAPI):
    @staticmethod
    def from_env():
        project = os.environ['PROJECT']
        zone = os.environ['ZONE'].rsplit('/', 1)[1]
        return GCPWorkerAPI(project, zone)

    def __init__(self, project: str, zone: str):
        self.project = project
        self.zone = zone

    @property
    def nameserver_ip(self):
        return '169.254.169.254'

    def create_disk(self, instance_name: str, disk_name: str, size_in_gb: int, mount_path: str) -> GCPDisk:
        return GCPDisk(
            zone=self.zone,
            project=self.project,
            instance_name=instance_name,
            name=disk_name,
            size_in_gb=size_in_gb,
            mount_path=mount_path,
        )

    def user_credentials(self, credentials: Dict[str, bytes]) -> GCPUserCredentials:
        return GCPUserCredentials(credentials)

    async def worker_access_token(self, session: httpx.ClientSession) -> Dict[str, str]:
        async with await request_retry_transient_errors(
            session,
            'POST',
            'http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token',
            headers={'Metadata-Flavor': 'Google'},
            timeout=aiohttp.ClientTimeout(total=60),  # type: ignore
        ) as resp:
            access_token = (await resp.json())['access_token']
            return {'username': 'oauth2accesstoken', 'password': access_token}

    def instance_config_from_config_dict(self, config_dict: Dict[str, str]) -> GCPSlimInstanceConfig:
        return GCPSlimInstanceConfig.from_dict(config_dict)

    def __str__(self):
        return f'project={self.project} zone={self.zone}'
