from typing import Any, Dict, TYPE_CHECKING

from .gcp.instance_config import GCPInstanceConfig


if TYPE_CHECKING:
    from ..instance_config import InstanceConfig  # pylint: disable=cyclic-import
    from ..inst_coll_config import PoolConfig  # pylint: disable=cyclic-import


def instance_config_from_config_dict(config: Dict[str, Any]) -> 'InstanceConfig':
    cloud = config.get('cloud', 'gcp')
    assert cloud == 'gcp'
    return GCPInstanceConfig(config)


def instance_config_from_pool_config(pool_config: 'PoolConfig') -> 'InstanceConfig':
    cloud = pool_config.cloud
    assert cloud == 'gcp'
    return GCPInstanceConfig.from_instance_properties(pool_config.boot_disk_size_gb,
                                                      pool_config.worker_local_ssd_data_disk,
                                                      pool_config.worker_pd_ssd_data_disk_size_gb,
                                                      pool_config.worker_type,
                                                      pool_config.worker_cores)
