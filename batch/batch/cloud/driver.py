import os
from typing import List

from gear import Database
from gear.cloud_config import get_global_config

from ..driver.driver import CloudDriver
from ..inst_coll_config import InstanceCollectionConfigs
from .azure.driver.driver import AzureDriver
from .gcp.driver.driver import GCPDriver
from .lambda_.driver.driver import LambdaDriver
from .terra.azure.driver.driver import TerraAzureDriver


async def get_cloud_drivers(
    app,
    db: Database,
    machine_name_prefix: str,
    namespace: str,
    inst_coll_configs: InstanceCollectionConfigs,
) -> List[CloudDriver]:
    drivers = []

    clouds = get_global_config()['cloud'].split(',')
    for cloud in clouds:
        if cloud == 'azure':
            if os.environ.get('HAIL_TERRA'):
                driver = await TerraAzureDriver.create(app, db, machine_name_prefix, namespace, inst_coll_configs, cloud)
            else:
                driver = await AzureDriver.create(app, db, machine_name_prefix, namespace, inst_coll_configs, cloud)
        elif cloud == 'lambda':
            driver = await LambdaDriver.create(app, db, machine_name_prefix, namespace, inst_coll_configs, cloud)
        else:
            assert cloud == 'gcp', cloud
            driver = await GCPDriver.create(app, db, machine_name_prefix, namespace, inst_coll_configs, cloud)

        drivers.append(driver)

    return drivers
