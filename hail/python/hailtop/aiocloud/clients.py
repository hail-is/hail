from typing import Optional

from gear.cloud_config import get_gcp_config, get_global_config

from . import aiogoogle, aioazure


def get_identity_client(credentials_file: Optional[str] = None):
    # FIXME: rename '/gsa-key/key.json' with a name that is cloud-agnostic
    if credentials_file is None:
        credentials_file = '/gsa-key/key.json'

    cloud = get_global_config['cloud']
    if cloud == 'gcp':
        project = get_gcp_config().project
        return aiogoogle.GoogleIAmClient(
            project, credentials=aiogoogle.GoogleCredentials.from_file(credentials_file)
        )

    assert cloud == 'azure'
    return aioazure.AzureGraphClient(
        credentials=aioazure.AzureCredentials.from_file(credentials_file),
        scopes=['https://graph.microsoft.com/.default']
    )
