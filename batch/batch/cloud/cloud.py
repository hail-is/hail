from typing import Dict, TYPE_CHECKING

from gear.cloud_config import get_global_config, get_gcp_config

from .gcp.driver.driver import GCPDriver
from .gcp.worker.disk import GCPDisk
from .gcp.worker.credentials import GCPUserCredentials


if TYPE_CHECKING:
    from ..inst_coll_config import InstanceCollectionConfigs  # pylint: disable=cyclic-import
    from ..worker.disk import CloudDisk  # pylint: disable=cyclic-import
    from ..worker.credentials import CloudUserCredentials  # pylint: disable=cyclic-import
    from ..driver.driver import CloudDriver  # pylint: disable=cyclic-import


def get_cloud_disk(instance_name: str, disk_name: str, size_in_gb: int, mount_path: str) -> 'CloudDisk':
    cloud = get_global_config()['cloud']
    assert cloud == 'gcp'
    gcp_config = get_gcp_config()
    disk = GCPDisk(
        zone=gcp_config.zone,
        project=gcp_config.project,
        instance_name=instance_name,
        name=disk_name,
        size_in_gb=size_in_gb,
        mount_path=mount_path,
    )
    return disk


def get_user_credentials(credentials: Dict[str, bytes]) -> 'CloudUserCredentials':
    cloud = get_global_config()['cloud']
    assert cloud == 'gcp'
    return GCPUserCredentials(credentials)


async def get_cloud_driver(app, machine_name_prefix: str, inst_coll_configs: 'InstanceCollectionConfigs', credentials_file: str) -> 'CloudDriver':
    cloud = get_global_config()['cloud']
    assert cloud == 'gcp'
    return await GCPDriver.create(app, machine_name_prefix, inst_coll_configs, credentials_file)
