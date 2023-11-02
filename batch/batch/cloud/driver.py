from gear import Database
from gear.cloud_config import get_global_config

from ..driver.driver import CloudDriver
from ..inst_coll_config import InstanceCollectionConfigs
from .azure.driver.driver import AzureDriver
from .gcp.driver.driver import GCPDriver


async def get_cloud_driver(
    app,
    db: Database,
    machine_name_prefix: str,
    namespace: str,
    inst_coll_configs: InstanceCollectionConfigs,
) -> CloudDriver:
    cloud = get_global_config()['cloud']

    if cloud == 'azure':
        return await AzureDriver.create(app, db, machine_name_prefix, namespace, inst_coll_configs)

    assert cloud == 'gcp', cloud
    return await GCPDriver.create(app, db, machine_name_prefix, namespace, inst_coll_configs)
