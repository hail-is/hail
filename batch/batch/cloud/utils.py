from typing import Any, Dict, Set
import os

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
        azure_config = get_azure_config()
        return {azure_config.region}
    assert cloud == 'gcp'
    gcp_config = get_gcp_config()
    return gcp_config.regions


ACCEPTABLE_QUERY_JAR_URL_PREFIX = (
    os.environ['HAIL_QUERY_STORAGE_URI'] + os.environ['HAIL_QUERY_ACCEPTABLE_JAR_SUBFOLDER']
)
assert len(ACCEPTABLE_QUERY_JAR_URL_PREFIX) > 3  # x:// where x is one or more characters
