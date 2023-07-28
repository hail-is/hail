import asyncio
import os
from typing import List, Optional, Tuple
from typing_extensions import Annotated as Ann
import typer

from typer import Abort, Exit, Option as Opt
# from rich import print
from rich.prompt import Confirm, IntPrompt, Prompt


async def setup_existing_remote_tmpdir(expected_region: str, service_account: str, verbose: bool) -> Tuple[Optional[str], bool]:
    from hailtop.aiogoogle import GoogleStorageAsyncFS  # pylint: disable=import-outside-toplevel

    from .utils import (
        InsufficientPermissions,
        get_gcp_bucket_information,
        grant_service_account_bucket_read_access,
        grant_service_account_bucket_write_access,
    )  # pylint: disable=import-outside-toplevel

    warnings = False

    remote_tmpdir = Prompt.ask(f'Enter a path to an existing remote temporary directory (ex: gs://my-bucket/batch/tmp)')

    bucket, _ = GoogleStorageAsyncFS.get_bucket_and_name(remote_tmpdir)

    try:
        bucket_info = await get_gcp_bucket_information(bucket, verbose)
    except InsufficientPermissions as e:
        typer.secho(e.message, fg=typer.colors.RED)
        raise Abort()

    location = bucket_info['location'].lower()
    if location != expected_region or bucket_info['locationType'] != 'region':
        typer.secho(f'WARNING: remote temporary directory {remote_tmpdir} is not located in {expected_region} or is multi-regional. '
                    f'Found {location} with location type {bucket_info["locationType"]}.',
                    fg=typer.colors.MAGENTA)
        warnings = True

    storage_class = bucket_info['storageClass']
    if storage_class.upper() != 'STANDARD':
        typer.secho(f'WARNING: remote temporary directory {remote_tmpdir} does not have storage class "STANDARD". Excess data charges will occur.',
                    fg=typer.colors.MAGENTA)
        warnings = True

    give_access_to_remote_tmpdir = Confirm.ask(
        f'Do you want to give service account {service_account} read/write access to bucket {bucket}?', default=True
    )
    if give_access_to_remote_tmpdir:
        try:
            await grant_service_account_bucket_read_access(project=None, bucket=bucket, service_account=service_account, verbose=verbose)
            await grant_service_account_bucket_write_access(project=None, bucket=bucket, service_account=service_account, verbose=verbose)
        except InsufficientPermissions as e:
            typer.secho(e.message, fg=typer.colors.RED)
            raise Abort()
        else:
            typer.secho(f'Granted service account {service_account} read and write access to {bucket}.', fg=typer.colors.GREEN)

    return (remote_tmpdir, warnings)


async def setup_new_remote_tmpdir(*,
                                  bucket_name: str,
                                  region: str,
                                  project: str,
                                  username: str,
                                  service_account: str,
                                  verbose: bool) -> Tuple[Optional[str], bool]:
    from .utils import (
        InsufficientPermissions,
        create_gcp_bucket,
        grant_service_account_bucket_read_access,
        grant_service_account_bucket_write_access,
    )  # pylint: disable=import-outside-toplevel

    remote_tmpdir = f'gs://{bucket_name}/batch/tmp'
    warnings = False

    try:
        set_lifecycle = Confirm.ask(
            f'Do you want to set a lifecycle policy (automatically delete files after a time period) on the bucket {bucket_name}?',
            default=True)
        if set_lifecycle:
            lifecycle_days = IntPrompt.ask(
                f'After how many days should files be automatically deleted from bucket {bucket_name}?', default=30)
            if lifecycle_days <= 0:
                typer.secho(f'invalid value for lifecycle rule in days {lifecycle_days}', fg=typer.colors.RED)
                raise Abort()
        else:
            lifecycle_days = None

        labels = {
            'bucket': bucket_name,
            'owner': username,
            'data_type': 'temporary',
        }
        await create_gcp_bucket(
            project=project,
            bucket=bucket_name,
            location=region,
            lifecycle_days=lifecycle_days,
            labels=labels,
            verbose=verbose,
        )
        typer.secho(f'Created bucket {bucket_name} in project {project} with lifecycle rule set to {lifecycle_days} days.', fg=typer.colors.GREEN)
    except InsufficientPermissions as e:
        typer.secho(e.message, fg=typer.colors.RED)
        raise Abort()
    else:
        try:
            await grant_service_account_bucket_read_access(project, bucket_name, service_account, verbose=verbose)
            await grant_service_account_bucket_write_access(project, bucket_name, service_account, verbose=verbose)
        except InsufficientPermissions as e:
            typer.secho(e.message, fg=typer.colors.RED)
            raise Abort()
        else:
            typer.secho(f'Granted service account {service_account} read and write access to {bucket_name} in project {project}.',
                        fg=typer.colors.GREEN)

    return (remote_tmpdir, warnings)


async def initialize_gcp(username: str,
                         hail_identity: str,
                         supported_regions: List[str],
                         default_region: str,
                         verbose: bool) -> Tuple[Optional[str], str, bool]:
    from hailtop.utils import secret_alnum_string  # pylint: disable=import-outside-toplevel

    from .utils import check_for_gcloud, get_gcp_default_project, get_default_region  # pylint: disable=import-outside-toplevel

    gcloud_installed = await check_for_gcloud()
    if not gcloud_installed:
        typer.secho(f'Have you installed gcloud? For directions see https://cloud.google.com/sdk/docs/install '
                    f'Are you logged into your google account by running `gcloud auth login`?',
                    fg=typer.colors.RED)
        raise Abort()

    default_project = await get_gcp_default_project(verbose=verbose)
    project = Prompt.ask('Which google project should resources be created in?', default=default_project)

    region = Prompt.ask('Which region should resources be created in?', default=default_region, choices=supported_regions)

    token = secret_alnum_string(5).lower()
    maybe_bucket_name = f'hail-batch-{username}-{token}'

    create_remote_tmpdir = Confirm.ask(f'Do you want to create a new bucket "{maybe_bucket_name}" in project "{project}" for temporary files generated by Hail?',
                                       default=True)
    if create_remote_tmpdir:
        remote_tmpdir, warnings = await setup_new_remote_tmpdir(
            bucket_name=maybe_bucket_name,
            region=region,
            project=project,
            username=username,
            service_account=hail_identity,
            verbose=verbose
        )
    else:
        remote_tmpdir, warnings = await setup_existing_remote_tmpdir(region, hail_identity, verbose)

    return (remote_tmpdir, region, warnings)


async def async_basic_initialize(incremental: bool = False, verbose: bool = False):
    from hailtop.auth.auth import async_get_userinfo  # pylint: disable=import-outside-toplevel
    from hailtop.batch_client.aioclient import BatchClient  # pylint: disable=import-outside-toplevel
    from hailtop.config import DeployConfig  # pylint: disable=import-outside-toplevel
    from hailtop.config import get_user_config_path  # pylint: disable=import-outside-toplevel

    from ..auth.login import async_login  # pylint: disable=import-outside-toplevel
    from .cli import set as set_config, list as list_config  # pylint: disable=import-outside-toplevel
    from .utils import already_logged_into_service, get_default_region  # pylint: disable=import-outside-toplevel

    domain = Prompt.ask('What domain is the Hail service running in?', default='hail.is')
    namespace = 'default'

    deploy_config = DeployConfig('external', namespace, domain)
    deploy_config.dump_to_file()

    already_logged_in = await already_logged_into_service()
    if not already_logged_in:
        await async_login(namespace)

    user_info = await async_get_userinfo()
    username = user_info['username']
    hail_identity = user_info['hail_identity']
    trial_bp_name = user_info['trial_bp_name']

    batch_client = await BatchClient.create(trial_bp_name, deploy_config=deploy_config)

    async with batch_client:
        cloud = await batch_client.cloud()
        supported_regions = await batch_client.supported_regions()
        default_region = get_default_region(supported_regions, 'gcp')

        if cloud == 'gcp':
            remote_tmpdir, region, warnings = await initialize_gcp(username, hail_identity, supported_regions, default_region, verbose)
        else:
            region = Prompt.ask('Which region should resources be created in?', default=default_region,
                                choices=supported_regions)

            remote_tmpdir = Prompt.ask(f'Enter a path to an existing remote temporary directory (ex: gs://my-bucket/batch/tmp)')
            typer.secho(f'WARNING: You will need to grant read/write access to {remote_tmpdir} for account {hail_identity}', fg=typer.colors.MAGENTA)
            warnings = True

    config_file = get_user_config_path()

    if not incremental:
        if os.path.isfile(config_file):
            os.remove(config_file)

    set_config('domain', domain)

    if trial_bp_name:
        set_config('batch/billing_project', trial_bp_name)

    if remote_tmpdir:
        set_config('batch/remote_tmpdir', remote_tmpdir)

    set_config('batch/regions', region)
    set_config('batch/backend', 'service')
    set_config('query/backend', 'batch')

    typer.secho('--------------------', fg=typer.colors.BLUE)
    typer.secho('FINAL CONFIGURATION:', fg=typer.colors.BLUE)
    typer.secho('--------------------', fg=typer.colors.BLUE)
    list_config()

    if warnings:
        typer.secho('WARNING: Initialized Hail with warnings! The currently specified configuration causes excess costs when using Hail Batch.',
                    fg=typer.colors.MAGENTA)
        raise Exit()


def initialize(
        incremental: Ann[bool, Opt('--incremental', '-i', help='Do not destroy the existing configuration before setting new variables.')] = False,
        verbose: Ann[bool, Opt('--verbose', '-v', help='Print gcloud commands being executed')] = False
):
    asyncio.get_event_loop().run_until_complete(async_basic_initialize(incremental=incremental, verbose=verbose))
