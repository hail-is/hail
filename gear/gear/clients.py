from typing import Optional

from gear.cloud_config import get_azure_config, get_gcp_config, get_global_config
from hailtop.aiocloud import aioazure, aiogoogle, common
from hailtop.aiotools.fs import AsyncFS, AsyncFSFactory


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


def get_compute_client(*args, **kwargs):
    if kwargs.get('credentials_file') is None and kwargs.get('credentials') is None:
        kwargs['credentials_file'] = '/gsa-key/key.json'

    cloud = get_global_config()['cloud']

    if cloud == 'azure':
        azure_config = get_azure_config()
        return aioazure.AzureComputeClient(azure_config.subscription_id, azure_config.resource_group, *args, **kwargs)

    assert cloud == 'gcp', cloud
    project = get_gcp_config().project
    return aiogoogle.GoogleComputeClient(project, *args, **kwargs)


def get_cloud_credentials_from_data(data, *args, **kwargs) -> common.CloudCredentials:
    cloud = get_global_config()['cloud']

    if cloud == 'azure':
        return aioazure.AzureCredentials.from_credentials_data(data, *args, **kwargs)

    assert cloud == 'gcp', cloud
    return aiogoogle.GoogleCredentials.from_credentials_data(data, *args, **kwargs)


def get_cloud_async_fs(*args, **kwargs) -> AsyncFS:
    if kwargs.get('credentials_file') is None and kwargs.get('credentials') is None:
        kwargs['credentials_file'] = '/gsa-key/key.json'

    cloud = get_global_config()['cloud']

    if cloud == 'azure':
        return aioazure.AzureAsyncFS(*args, **kwargs)

    assert cloud == 'gcp', cloud
    return aiogoogle.GoogleStorageAsyncFS(*args, **kwargs)


def get_cloud_async_fs_factory() -> AsyncFSFactory:
    cloud = get_global_config()['cloud']
    if cloud == 'azure':
        return aioazure.AzureAsyncFSFactory()
    assert cloud == 'gcp', cloud
    return aiogoogle.GoogleStorageAsyncFSFactory()
