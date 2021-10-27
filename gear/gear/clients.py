from typing import Optional

from gear.cloud_config import get_gcp_config, get_global_config

from hailtop.aiocloud import aiogoogle, aioazure
from hailtop.aiotools.fs import AsyncFS


def get_identity_client(credentials_file: Optional[str] = None):
    if credentials_file is None:
        credentials_file = '/gsa-key/key.json'

    cloud = get_global_config()['cloud']

    if cloud == 'gcp':
        project = get_gcp_config().project
        return aiogoogle.GoogleIAmClient(project, credentials_file=credentials_file)

    assert cloud == 'azure'
    scopes = ['https://graph.microsoft.com/.default']
    return aioazure.AzureGraphClient(
        credentials_file=credentials_file,
        scopes=scopes,
    )


def get_compute_client(credentials_file: Optional[str] = None):
    if credentials_file is None:
        credentials_file = '/gsa-key/key.json'

    cloud = get_global_config()['cloud']

    assert cloud == 'gcp', cloud
    project = get_gcp_config().project
    return aiogoogle.GoogleComputeClient(project, credentials_file=credentials_file)


def get_cloud_async_fs(credentials_file: Optional[str] = None) -> AsyncFS:
    if credentials_file is None:
        credentials_file = '/gsa-key/key.json'

    cloud = get_global_config()['cloud']

    assert cloud == 'gcp', cloud
    project = get_gcp_config().project
    return aiogoogle.GoogleStorageAsyncFS(project=project, credentials_file=credentials_file)
