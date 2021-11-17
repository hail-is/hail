from typing import Any, Dict

from ..instance_config import InstanceConfig
from .azure.instance_config import AzureSlimInstanceConfig
from .gcp.instance_config import GCPSlimInstanceConfig


def instance_config_from_config_dict(config: Dict[str, Any]) -> InstanceConfig:
    cloud = config.get('cloud', 'gcp')
    if cloud == 'azure':
        return AzureSlimInstanceConfig.from_dict(config)
    assert cloud == 'gcp'
    return GCPSlimInstanceConfig.from_dict(config)
