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
    return GCPInstanceConfig.from_pool_config(pool_config)
