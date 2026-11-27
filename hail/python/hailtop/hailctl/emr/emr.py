from typing import Optional

import typer

from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.utils import async_to_blocking


def emr_client(region: Optional[str]):
    """Create an EMR client.

    If `region` is None, boto3 resolves it the way the AWS CLI does: from the
    AWS_REGION / AWS_DEFAULT_REGION environment variables or the profile set by
    `aws configure`.
    """
    import boto3  # pylint: disable=import-outside-toplevel

    session = boto3.session.Session(region_name=region)
    if session.region_name is None:
        raise typer.BadParameter(
            'Could not determine the AWS region. Use --region, or run `aws configure set region <region>`.'
        )
    return session.client('emr')


def upload_to_s3(dest_uri: str, data: bytes) -> None:
    """Write bytes to an s3:// URI through Hail's RouterAsyncFS."""

    async def _upload() -> None:
        async with RouterAsyncFS() as fs:
            await fs.write(dest_uri, data)

    async_to_blocking(_upload())
