from typing import Dict

import aiohttp

from .azure.worker.credentials import AzureUserCredentials
from .azure.worker.disk import AzureDisk
from .azure.worker.instance_env import AzureInstanceEnvironment
from .azure.worker.utils import azure_worker_access_token

from .gcp.worker.credentials import GCPUserCredentials
from .gcp.worker.disk import GCPDisk
from .gcp.worker.instance_env import GCPInstanceEnvironment
from .gcp.worker.utils import gcp_worker_access_token

from ..worker.credentials import CloudUserCredentials
from ..worker.disk import CloudDisk
from ..worker.instance_env import CloudInstanceEnvironment


def get_instance_environment(cloud: str):
    if cloud == 'azure':
        return AzureInstanceEnvironment.from_env()
    assert cloud == 'gcp', cloud
    return GCPInstanceEnvironment.from_env()


def get_cloud_disk(instance_environment: CloudInstanceEnvironment,
                   instance_name: str,
                   disk_name: str,
                   size_in_gb: int,
                   mount_path: str,
                   ) -> CloudDisk:
    if isinstance(instance_environment, AzureInstanceEnvironment):
        return AzureDisk(disk_name, instance_name, size_in_gb, mount_path)

    assert isinstance(instance_environment, GCPInstanceEnvironment)
    return GCPDisk(
        zone=instance_environment.zone,
        project=instance_environment.project,
        instance_name=instance_name,
        name=disk_name,
        size_in_gb=size_in_gb,
        mount_path=mount_path,
    )


def get_user_credentials(cloud: str, credentials: Dict[str, bytes]) -> CloudUserCredentials:
    if cloud == 'azure':
        return AzureUserCredentials(credentials)
    assert cloud == 'gcp', cloud
    return GCPUserCredentials(credentials)


async def get_worker_access_token(cloud: str, session: aiohttp.ClientSession) -> Dict[str, str]:
    if cloud == 'azure':
        return await azure_worker_access_token(session)
    assert cloud == 'gcp', cloud
    return await gcp_worker_access_token(session)
