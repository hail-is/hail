from typing import Optional

from gear.cloud_config import get_azure_config, get_gcp_config, get_global_config

from hailtop.aiocloud import aiogoogle, aioazure
from hailtop.aiotools.fs import AsyncFS


def get_identity_client(credentials_file: Optional[str] = None):
    if credentials_file is None:
        credentials_file = '/gsa-key/key.json'

    cloud = get_global_config()['cloud']

    if cloud == 'azure':
        scopes = ['https://graph.microsoft.com/.default']
        return aioazure.AzureGraphClient(
            credentials_file=credentials_file,
            scopes=scopes,
        )

    assert cloud == 'gcp', cloud
    project = get_gcp_config().project
    return aiogoogle.GoogleIAmClient(project, credentials_file=credentials_file)


def get_compute_client(credentials_file: Optional[str] = None):
    if credentials_file is None:
        credentials_file = '/gsa-key/key.json'

    cloud = get_global_config()['cloud']

    if cloud == 'azure':
        azure_config = get_azure_config()
        return aioazure.AzureComputeClient(azure_config.subscription_id, azure_config.resource_group)

    assert cloud == 'gcp', cloud
    project = get_gcp_config().project
    return aiogoogle.GoogleComputeClient(project, credentials_file=credentials_file)


def get_cloud_async_fs(credentials_file: Optional[str] = None) -> AsyncFS:
    if credentials_file is None:
        credentials_file = '/gsa-key/key.json'

    cloud = get_global_config()['cloud']

    if cloud == 'azure':
        return aioazure.AzureAsyncFS(credential_file=credentials_file)

    assert cloud == 'gcp', cloud
    return aiogoogle.GoogleStorageAsyncFS(credentials_file=credentials_file)
