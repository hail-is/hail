from typing import Dict
import os

from .gcp.worker.credentials import GCPUserCredentials
from .gcp.worker.disk import GCPDisk

from ..worker.credentials import CloudUserCredentials
from ..worker.disk import CloudDisk


class CloudInstanceEnvironment:
    pass


class GCPInstanceEnvironment(CloudInstanceEnvironment):
    @staticmethod
    def from_env():
        project = os.environ['PROJECT']
        zone = os.environ['ZONE'].rsplit('/', 1)[1]
        return GCPInstanceEnvironment(project, zone)

    def __init__(self, project: str, zone: str):
        self.project = project
        self.zone = zone

    def __str__(self):
        return f'project={self.project} zone={self.zone}'


def get_instance_environment(cloud: str):
    assert cloud == 'gcp', cloud
    return GCPInstanceEnvironment.from_env()


def get_cloud_disk(instance_environment: CloudInstanceEnvironment,
                   instance_name: str,
                   disk_name: str,
                   size_in_gb: int,
                   mount_path: str,
                   ) -> CloudDisk:
    assert isinstance(instance_environment, GCPInstanceEnvironment)
    disk = GCPDisk(
        zone=instance_environment.zone,
        project=instance_environment.project,
        instance_name=instance_name,
        name=disk_name,
        size_in_gb=size_in_gb,
        mount_path=mount_path,
    )
    return disk


def get_user_credentials(cloud: str, credentials: Dict[str, bytes]) -> CloudUserCredentials:
    assert cloud == 'gcp'
    return GCPUserCredentials(credentials)
