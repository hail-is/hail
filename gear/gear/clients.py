from gear.cloud_config import get_gcp_config, get_global_config
from hailtop.aiocloud import aioazure, aiogoogle, aioaws
from hailtop.aiocloud.aioterra import azure as aioterra_azure
from hailtop.aiotools.fs import AsyncFS

import boto3

def get_identity_client():
    cloud = get_global_config()['cloud']

    if cloud == 'azure':
        return aioazure.AzureGraphClient()
    elif cloud == 'aws':
        return boto3.client('iam')

    assert cloud == 'gcp', cloud
    project = get_gcp_config().project
    return aiogoogle.GoogleIAmClient(project)


def get_cloud_async_fs() -> AsyncFS:
    cloud = get_global_config()['cloud']

    if cloud == 'azure':
        if aioterra_azure.TerraAzureAsyncFS.enabled():
            return aioterra_azure.TerraAzureAsyncFS()
        return aioazure.AzureAsyncFS()
    elif cloud == 'aws':
        return aioaws.S3AsyncFS()

    assert cloud == 'gcp', cloud
    return aiogoogle.GoogleStorageAsyncFS()
