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


DEFAULT_SERVICE_ROLE = 'EMR_DefaultRole'
DEFAULT_JOB_FLOW_ROLE = 'EMR_EC2_DefaultRole'


def _role_exists(iam, role_name: str) -> bool:
    from botocore.exceptions import ClientError  # pylint: disable=import-outside-toplevel

    try:
        iam.get_role(RoleName=role_name)
        return True
    except ClientError as exc:
        if exc.response.get('Error', {}).get('Code') == 'NoSuchEntity':
            return False
        raise


def check_default_roles(iam=None) -> None:
    """Verify the EMR default IAM roles exist, printing a clear message.

    `aws emr create-default-roles` prints only the roles it *creates*, so it
    returns an empty list (``[]``) when they already exist -- which reads as if
    nothing happened. This preflight instead reports explicitly that the roles
    are present, and raises an actionable error listing any that are missing.

    IAM is global, so no region is needed.
    """
    if iam is None:
        import boto3  # pylint: disable=import-outside-toplevel

        iam = boto3.client('iam')

    roles = (DEFAULT_SERVICE_ROLE, DEFAULT_JOB_FLOW_ROLE)
    missing = [role for role in roles if not _role_exists(iam, role)]
    if missing:
        raise ValueError(
            f"Missing EMR default IAM role(s): {', '.join(missing)}. "
            f"Create them once with `aws emr create-default-roles`, or pass "
            f"--no-use-default-roles together with --service-role and --instance-profile."
        )
    print(f"Using existing EMR default roles: {', '.join(roles)}.")


def upload_to_s3(dest_uri: str, data: bytes) -> None:
    """Write bytes to an s3:// URI through Hail's RouterAsyncFS.

    S3 file I/O goes through the same FS abstraction the rest of hailtop uses,
    rather than a raw boto3 S3 client.
    """

    async def _upload() -> None:
        async with RouterAsyncFS() as fs:
            await fs.write(dest_uri, data)

    async_to_blocking(_upload())
