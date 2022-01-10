import aiohttp
from typing import Dict
import os
import json
import base64

from hailtop.utils import request_retry_transient_errors
from hailtop import httpx

from ....worker.instance_env import CloudWorkerAPI
from .disk import GCPDisk, GCPDiskManager
from .credentials import GCPUserCredentials
from ..instance_config import GCPSlimInstanceConfig


class GCPWorkerAPI(CloudWorkerAPI):
    @staticmethod
    def from_env():
        project = os.environ['PROJECT']
        zone = os.environ['ZONE'].rsplit('/', 1)[1]
        disk_manager = GCPDiskManager(project, zone)
        instance_config = GCPSlimInstanceConfig.from_dict(
            json.loads(base64.b64decode(os.environ['INSTANCE_CONFIG']).decode())
        )

        return GCPWorkerAPI(project, zone, disk_manager, instance_config)

    def __init__(self, project: str, zone: str, disk_manager: GCPDiskManager, instance_config: GCPSlimInstanceConfig):
        self.project = project
        self.zone = zone
        self.disk_manager = disk_manager
        self.instance_config = instance_config

    @property
    def nameserver_ip(self):
        return '169.254.169.254'

    async def new_disk(self, instance_name: str, disk_name: str, size_in_gb: int, mount_path: str) -> GCPDisk:
        return await self.disk_manager.new_disk(instance_name, disk_name, size_in_gb, mount_path)

    async def delete_disk(self, disk: GCPDisk):  # type: ignore[override]
        return await self.disk_manager.delete_disk(disk)

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

    def __str__(self):
        return f'project={self.project} zone={self.zone}'
