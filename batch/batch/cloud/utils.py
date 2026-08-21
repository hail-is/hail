import os
from typing import Any, Dict, Set

from gear.cloud_config import get_azure_config, get_gcp_config
from hailtop.aiocloud.aioazure import AzureAsyncFS
from hailtop.aiocloud.aiogoogle import GoogleStorageAsyncFS

from ..instance_config import InstanceConfig
from .azure.instance_config import AzureSlimInstanceConfig
from .gcp.instance_config import GCPSlimInstanceConfig
from .terra.azure.instance_config import TerraAzureSlimInstanceConfig


def instance_config_from_config_dict(config: Dict[str, Any]) -> InstanceConfig:
    cloud = config.get('cloud', 'gcp')
    if cloud == 'azure':
        if os.environ.get('HAIL_TERRA'):
            return TerraAzureSlimInstanceConfig.from_dict(config)
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


def _query_storage_url_prefix(subfolder_envvar: str) -> str:
    cloud = os.environ['CLOUD']
    query_storage_uri = os.environ['HAIL_QUERY_STORAGE_URI']
    subfolder = os.environ[subfolder_envvar]
    url_prefix = query_storage_uri + subfolder

    assert subfolder[0] == '/', (query_storage_uri, subfolder)
    assert query_storage_uri[-1] != '/', (query_storage_uri, subfolder)

    if cloud == 'gcp':
        assert GoogleStorageAsyncFS.valid_url(url_prefix)
    else:
        assert cloud == 'azure'
        assert AzureAsyncFS.valid_url(url_prefix)

    return url_prefix


ACCEPTABLE_QUERY_JAR_URL_PREFIX = _query_storage_url_prefix('HAIL_QUERY_ACCEPTABLE_JAR_SUBFOLDER')
SPARK_ARCHIVE_URL_PREFIX = _query_storage_url_prefix('HAIL_QUERY_SPARK_ARCHIVE_SUBFOLDER')
