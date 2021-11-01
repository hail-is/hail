from typing import Dict

from .gcp.worker.credentials import GCPUserCredentials
from .gcp.worker.disk import GCPDisk
from .gcp.instance_config import GCPSlimInstanceConfig

from ..worker.credentials import CloudUserCredentials
from ..worker.disk import CloudDisk
from ..instance_config import InstanceConfig


def get_cloud_disk(location: str,
                   project: str,
                   instance_name: str,
                   disk_name: str,
                   size_in_gb: int,
                   mount_path: str,
                   instance_config: InstanceConfig
                   ) -> CloudDisk:
    cloud = instance_config.cloud
    assert cloud == 'gcp'
    assert isinstance(instance_config, GCPSlimInstanceConfig)
    disk = GCPDisk(
        zone=location,
        project=project,
        instance_name=instance_name,
        name=disk_name,
        size_in_gb=size_in_gb,
        mount_path=mount_path,
    )
    return disk


def get_user_credentials(cloud: str, credentials: Dict[str, bytes]) -> CloudUserCredentials:
    assert cloud == 'gcp'
    return GCPUserCredentials(credentials)
