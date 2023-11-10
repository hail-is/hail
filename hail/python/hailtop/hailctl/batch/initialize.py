from typing import List, Optional, Tuple
import typer

from typer import Abort, Exit
from rich.prompt import Confirm, IntPrompt, Prompt

from hailtop.config import ConfigVariable


async def setup_existing_remote_tmpdir(service_account: str, verbose: bool) -> Tuple[Optional[str], str, bool]:
    from hailtop.aiogoogle import GoogleStorageAsyncFS  # pylint: disable=import-outside-toplevel

    from .utils import InsufficientPermissions, get_gcp_bucket_information, grant_service_account_bucket_access_with_role  # pylint: disable=import-outside-toplevel

    warnings = False

    remote_tmpdir = Prompt.ask('Enter a path to an existing remote temporary directory (ex: gs://my-bucket/batch/tmp)')

    bucket, _ = GoogleStorageAsyncFS.get_bucket_and_name(remote_tmpdir)

    try:
        bucket_info = await get_gcp_bucket_information(bucket, verbose)
    except InsufficientPermissions as e:
        typer.secho(e.message, fg=typer.colors.RED)
        raise Abort() from e

    location = bucket_info['location'].lower()

    if bucket_info['locationType'] != 'region':
        typer.secho(f'WARNING: remote temporary directory {remote_tmpdir} is multi-regional. Using this bucket with the Batch Service will incur additional network fees.',
                    fg=typer.colors.YELLOW)
        warnings = True

    storage_class = bucket_info['storageClass']
    if storage_class.upper() != 'STANDARD':
        typer.secho(f'WARNING: remote temporary directory {remote_tmpdir} does not have storage class "STANDARD". Additional data recovery charges will occur when accessing data.',
                    fg=typer.colors.YELLOW)
        warnings = True

    give_access_to_remote_tmpdir = Confirm.ask(f'Do you want to give service account {service_account} read/write access to bucket {bucket}?')
    if give_access_to_remote_tmpdir:
        try:
            await grant_service_account_bucket_access_with_role(bucket=bucket, service_account=service_account, role= 'roles/storage.objectAdmin', verbose=verbose)
        except InsufficientPermissions as e:
            typer.secho(e.message, fg=typer.colors.RED)
            raise Abort() from e
        typer.secho(f'Granted service account {service_account} read and write access to {bucket}.', fg=typer.colors.GREEN)
    else:
        typer.secho(f'WARNING: Please verify service account {service_account} has the role "roles/storage.objectAdmin" or '
                    f'both "roles/storage.objectViewer" and "roles/storage.objectCreator" roles for bucket {bucket}.',
                    fg=typer.colors.YELLOW)
        warnings = True

    return (remote_tmpdir, location, warnings)


async def setup_new_remote_tmpdir(*,
                                  supported_regions: List[str],
                                  username: str,
                                  service_account: str,
                                  verbose: bool) -> Tuple[Optional[str], str, bool]:
    from hailtop.utils import secret_alnum_string  # pylint: disable=import-outside-toplevel

    from .utils import BucketAlreadyExistsError, InsufficientPermissions, create_gcp_bucket, \
        get_gcp_default_project, grant_service_account_bucket_access_with_role, update_gcp_bucket  # pylint: disable=import-outside-toplevel

    token = secret_alnum_string(5).lower()
    maybe_bucket_name = f'hail-batch-{username}-{token}'
    bucket_name = Prompt.ask(f'What is the name of the new bucket (Example: {maybe_bucket_name})')

    default_project = await get_gcp_default_project(verbose=verbose)
    bucket_prompt = f'Which google project should {bucket_name} be created in? This project will incur costs for storing your Hail generated data.'
    if default_project is not None:
        bucket_prompt += f' (Example: {default_project})'
    project = Prompt.ask(bucket_prompt)

    if 'us-central1' in supported_regions:
        default_compute_region = 'us-central1'
    else:
        default_compute_region = supported_regions[0]

    bucket_region = Prompt.ask(f'Which region does your data reside in? (Example: {default_compute_region})')
    if bucket_region not in supported_regions:
        typer.secho(f'The region where your data lives ({bucket_region}) is not in one of the supported regions of the Batch Service ({supported_regions}). '
                    f'Creating a bucket in {bucket_region} will incur additional network fees when using the Batch Service.', fg=typer.colors.YELLOW)
        continue_w_region_error = Confirm.ask(f'Do you wish to continue setting up the new bucket {bucket_name} in region {bucket_region}?')
        if not continue_w_region_error:
            raise Abort()

    remote_tmpdir = f'gs://{bucket_name}/batch/tmp'
    warnings = False

    set_lifecycle = Confirm.ask(
        f'Do you want to set a lifecycle policy (automatically delete files after a time period) on the bucket {bucket_name}?')
    if set_lifecycle:
        lifecycle_days = IntPrompt.ask(
            f'After how many days should files be automatically deleted from bucket {bucket_name}?', default=30)
        if lifecycle_days <= 0:
            typer.secho(f'Invalid value for lifecycle rule in days {lifecycle_days}', fg=typer.colors.RED)
            raise Abort()
    else:
        lifecycle_days = None

    labels = {
        'bucket': bucket_name,
        'owner': username,
        'data_type': 'temporary',
    }

    try:
        await create_gcp_bucket(
            project=project,
            bucket=bucket_name,
            location=bucket_region,
            verbose=verbose,
        )
        typer.secho(f'Created bucket {bucket_name} in project {project}.', fg=typer.colors.GREEN)
    except InsufficientPermissions as e:
        typer.secho(e.message, fg=typer.colors.RED)
        raise Abort() from e
    except BucketAlreadyExistsError as e:
        typer.secho(e.message, fg=typer.colors.YELLOW)
        continue_w_update = Confirm.ask(f'Do you wish to continue updating the lifecycle rules and permissions on bucket {bucket_name}?')
        if not continue_w_update:
            typer.secho(f'WARNING: The lifecycle rules and permissions on bucket {bucket_name} were not updated. '
                        'You will have to manually configure these yourself.', fg=typer.colors.YELLOW)
            warnings = True
            return (remote_tmpdir, bucket_region, warnings)

    try:
        await update_gcp_bucket(
            project=project,
            bucket=bucket_name,
            lifecycle_days=lifecycle_days,
            labels=labels,
            verbose=verbose,
        )
    except InsufficientPermissions as e:
        typer.secho(e.message, fg=typer.colors.RED)
        raise Abort() from e

    typer.secho(f'Updated bucket {bucket_name} in project {project} with lifecycle rule set to {lifecycle_days} days and labels {labels}.', fg=typer.colors.GREEN)

    try:
        await grant_service_account_bucket_access_with_role(bucket_name, service_account, 'roles/storage.objectAdmin', verbose=verbose)
    except InsufficientPermissions as e:
        typer.secho(e.message, fg=typer.colors.RED)
        raise Abort() from e

    typer.secho(f'Granted service account {service_account} read and write access to {bucket_name} in project {project}.',
                fg=typer.colors.GREEN)

    return (remote_tmpdir, bucket_region, warnings)


async def initialize_gcp(username: str,
                         hail_identity: str,
                         supported_regions: List[str],
                         verbose: bool) -> Tuple[Optional[str], str, bool]:
    from .utils import check_for_gcloud  # pylint: disable=import-outside-toplevel
    assert len(supported_regions) > 0

    gcloud_installed = await check_for_gcloud()
    if not gcloud_installed:
        typer.secho('Have you installed gcloud? For directions see https://cloud.google.com/sdk/docs/install '
                    'To log into gcloud run:\n'
                    '> gcloud auth application-default login',
                    fg=typer.colors.RED)
        raise Abort()

    create_remote_tmpdir = Confirm.ask('Do you want to create a new bucket for temporary files generated by Hail?')
    if create_remote_tmpdir:
        remote_tmpdir, location, warnings = await setup_new_remote_tmpdir(
            supported_regions=supported_regions,
            username=username,
            service_account=hail_identity,
            verbose=verbose,
        )
    else:
        remote_tmpdir, location, warnings = await setup_existing_remote_tmpdir(hail_identity, verbose)

    return (remote_tmpdir, location, warnings)


async def async_basic_initialize(verbose: bool = False):
    from hailtop.auth import async_get_userinfo  # pylint: disable=import-outside-toplevel
    from hailtop.batch_client.aioclient import BatchClient  # pylint: disable=import-outside-toplevel
    from hailtop.config.deploy_config import get_deploy_config  # pylint: disable=import-outside-toplevel
    from hailtop.hailctl.config.cli import set as set_config, list as list_config  # pylint: disable=import-outside-toplevel

    from .utils import already_logged_into_service, login_to_service  # pylint: disable=import-outside-toplevel

    already_logged_in = await already_logged_into_service()
    if not already_logged_in:
        typer.secho('You are not currently logged in to Hail. Redirecting you to a login screen.', fg=typer.colors.YELLOW)
        await login_to_service()
        typer.secho('In the future, you can use `hailctl auth login` to login to Hail.', fg=typer.colors.YELLOW)
    else:
        domain = get_deploy_config()._domain
        typer.secho(f'You are currently logged in to Hail at domain {domain}.')

    user_info = await async_get_userinfo()
    assert user_info
    username = user_info['username']
    hail_identity = user_info['hail_identity']
    trial_bp_name = user_info['trial_bp_name']

    batch_client = await BatchClient.create(trial_bp_name)

    async with batch_client:
        cloud = await batch_client.cloud()
        supported_regions = await batch_client.supported_regions()

        if cloud == 'gcp':
            remote_tmpdir, tmpdir_region, warnings = await initialize_gcp(username, hail_identity, supported_regions, verbose)
        else:
            remote_tmpdir = Prompt.ask('Enter a path to an existing remote temporary directory (ex: https://myaccount.blob.core.windows.net/mycontainer/batch/tmp)')
            typer.secho(f'WARNING: You will need to grant read/write access to {remote_tmpdir} for account {hail_identity}', fg=typer.colors.YELLOW)
            warnings = False

            tmpdir_region = Prompt.ask('Which region is your remote temporary directory in? (Example: eastus)')

    compute_region = Prompt.ask('Which region do you want your jobs to run in?', choices=supported_regions)

    if tmpdir_region != compute_region and not compute_region.startswith(tmpdir_region):
        typer.secho(f'WARNING: remote temporary directory "{remote_tmpdir}" is not located in the selected compute region for Batch jobs "{compute_region}". Found {tmpdir_region}.',
                    fg=typer.colors.YELLOW)
        warnings = True

    if trial_bp_name:
        set_config(ConfigVariable.BATCH_BILLING_PROJECT, trial_bp_name)

    if remote_tmpdir:
        set_config(ConfigVariable.BATCH_REMOTE_TMPDIR, remote_tmpdir)

    set_config(ConfigVariable.BATCH_REGIONS, compute_region)

    set_config(ConfigVariable.BATCH_BACKEND, 'service')

    query_backend = Prompt.ask('Which backend do you want to use for Hail Query?', choices=['spark', 'batch'])
    set_config(ConfigVariable.QUERY_BACKEND, query_backend)

    typer.secho('--------------------', fg=typer.colors.BLUE)
    typer.secho('FINAL CONFIGURATION:', fg=typer.colors.BLUE)
    typer.secho('--------------------', fg=typer.colors.BLUE)
    list_config()

    if warnings:
        typer.secho('WARNING: The currently specified configuration will result in additional fees when using Hail Batch.',
                    fg=typer.colors.YELLOW)
        raise Exit()
