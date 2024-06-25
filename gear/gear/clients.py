from gear.cloud_config import get_gcp_config, get_global_config
from hailtop.aiocloud import aioazure, aiogoogle
from hailtop.aiocloud.aioterra import azure as aioterra_azure
from hailtop.aiotools.fs import AsyncFS, AsyncFSFactory


def get_identity_client():
    cloud = get_global_config()['cloud']

    if cloud == 'azure':
        return aioazure.AzureGraphClient()

    assert cloud == 'gcp', cloud
    project = get_gcp_config().project
    return aiogoogle.GoogleIAmClient(project)


def get_cloud_async_fs() -> AsyncFS:
    cloud = get_global_config()['cloud']

    if cloud == 'azure':
        if aioterra_azure.TerraAzureAsyncFS.enabled():
            return aioterra_azure.TerraAzureAsyncFS()
        return aioazure.AzureAsyncFS()

    assert cloud == 'gcp', cloud
    return aiogoogle.GoogleStorageAsyncFS()


def get_cloud_async_fs_factory() -> AsyncFSFactory:
    cloud = get_global_config()['cloud']
    if cloud == 'azure':
        return aioazure.AzureAsyncFSFactory()
    assert cloud == 'gcp', cloud
    return aiogoogle.GoogleStorageAsyncFSFactory()
