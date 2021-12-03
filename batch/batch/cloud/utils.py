from typing import Any, Dict, Set

from gear.cloud_config import get_azure_config, get_gcp_config

from ..instance_config import InstanceConfig
from .azure.instance_config import AzureSlimInstanceConfig
from .gcp.instance_config import GCPSlimInstanceConfig


def instance_config_from_config_dict(config: Dict[str, Any]) -> InstanceConfig:
    cloud = config.get('cloud', 'gcp')
    if cloud == 'azure':
        return AzureSlimInstanceConfig.from_dict(config)
    assert cloud == 'gcp'
    return GCPSlimInstanceConfig.from_dict(config)


def possible_cloud_locations(cloud: str) -> Set[str]:
    if cloud == 'azure':
        config = get_azure_config()
        return {config.region}
    assert cloud == 'gcp'
    config = get_gcp_config()
    return config.regions
