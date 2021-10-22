from typing import Optional

from gear.cloud_config import get_gcp_config, get_global_config

from hailtop.aiocloud import aiogoogle, aioazure


def get_identity_client(credentials_file: Optional[str] = None):
    if credentials_file is None:
        credentials_file = '/gsa-key/key.json'

    cloud = get_global_config()['cloud']
    if cloud == 'gcp':
        project = get_gcp_config().project
        return aiogoogle.GoogleIAmClient(project, credentials=aiogoogle.GoogleCredentials.from_file(credentials_file))

    assert cloud == 'azure'
    scopes = ['https://graph.microsoft.com/.default']
    return aioazure.AzureGraphClient(
        credentials=aioazure.AzureCredentials.from_file(credentials_file, scopes=scopes),
        scopes=scopes,
    )
