import os
from typing import Optional

from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.config import ConfigVariable, configuration_of
from hailtop.utils import async_to_blocking


def resolve_region(explicit_region: Optional[str]) -> Optional[str]:
    """Resolve the AWS region for EMR operations.

    Order: explicit argument, then the emr/region config variable, then the
    AWS_DEFAULT_REGION / AWS_REGION environment variables. Returns None if
    unset so that botocore can resolve it from the user's AWS config.
    """
    if explicit_region is not None:
        return explicit_region
    config_region = configuration_of(ConfigVariable.EMR_REGION, None, None)
    if config_region is not None:
        return config_region
    return os.environ.get('AWS_DEFAULT_REGION') or os.environ.get('AWS_REGION')


def emr_client(region: Optional[str]):
    import boto3  # pylint: disable=import-outside-toplevel

    return boto3.client('emr', region_name=region)


def upload_to_s3(dest_uri: str, data: bytes) -> None:
    """Write bytes to an s3:// URI through Hail's RouterAsyncFS.

    S3 file I/O goes through the same FS abstraction the rest of hailtop uses,
    rather than a raw boto3 S3 client.
    """

    async def _upload() -> None:
        async with RouterAsyncFS() as fs:
            await fs.write(dest_uri, data)

    async_to_blocking(_upload())
