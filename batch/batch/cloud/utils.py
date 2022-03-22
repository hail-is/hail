import os
from typing import Any, Dict, Set
from urllib.parse import urlparse

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


def _acceptable_query_jar_url_prefix() -> str:
    query_storage_uri = os.environ['HAIL_QUERY_STORAGE_URI']
    jar_subfolder = os.environ['HAIL_QUERY_ACCEPTABLE_JAR_SUBFOLDER']
    acceptable_query_jar_url_prefix = query_storage_uri + jar_subfolder

    assert jar_subfolder[0] == '/', (query_storage_uri, jar_subfolder)
    assert query_storage_uri[-1] != '/', (query_storage_uri, jar_subfolder)

    parsed = urlparse(acceptable_query_jar_url_prefix)
    assert parsed.scheme in {'hail-az', 'gs'}, (query_storage_uri, jar_subfolder)

    return acceptable_query_jar_url_prefix


ACCEPTABLE_QUERY_JAR_URL_PREFIX = _acceptable_query_jar_url_prefix()
