from typing import Any, Dict

from ..instance_config import InstanceConfig
from .gcp.instance_config import GCPSlimInstanceConfig


def instance_config_from_config_dict(config: Dict[str, Any]) -> InstanceConfig:
    cloud = config.get('cloud', 'gcp')
    assert cloud == 'gcp'
    return GCPSlimInstanceConfig.from_dict(config)
