from typing import TYPE_CHECKING

from gear.cloud_config import get_global_config

from .gcp.driver.driver import GCPDriver

if TYPE_CHECKING:
    from ..inst_coll_config import InstanceCollectionConfigs  # pylint: disable=cyclic-import
    from ..driver.driver import CloudDriver  # pylint: disable=cyclic-import


async def get_cloud_driver(app, machine_name_prefix: str, namespace: str, inst_coll_configs: 'InstanceCollectionConfigs',
                           credentials_file: str) -> 'CloudDriver':
    cloud = get_global_config()['cloud']
    assert cloud == 'gcp'
    return await GCPDriver.create(app, machine_name_prefix, namespace, inst_coll_configs, credentials_file)
