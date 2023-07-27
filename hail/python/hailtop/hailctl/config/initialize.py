import asyncio
import os
from typing import List, Optional, Tuple
import typer

from typer import Exit, Option as Opt
from rich.prompt import Confirm, IntPrompt, Prompt


async def setup_existing_remote_tmpdir(expected_region: str, service_account: str) -> Tuple[Optional[str], bool]:
    from hailtop.aiogoogle import GoogleStorageAsyncFS  # pylint: disable=import-outside-toplevel

    from .utils import (
        get_gcp_bucket_information,
        grant_service_account_bucket_read_access,
        grant_service_account_bucket_write_access,
    )  # pylint: disable=import-outside-toplevel

    errors = False
    remote_tmpdir = Prompt.ask(f'Enter a path to an existing remote temporary directory (ex: gs://my-bucket/batch/tmp)')

    async with GoogleStorageAsyncFS() as fs:
        bucket, _ = fs.get_bucket_and_name(remote_tmpdir)
        bucket_info = await get_gcp_bucket_information(fs._storage_client, bucket)
        if bucket_info.location != expected_region or bucket_info.is_multi_regional():
            print(f'WARNING: given remote temporary directory {remote_tmpdir} is not located in {expected_region} or is multi-regional. Found {bucket_info.location} with type {bucket_info.location_type}.')
            errors = True

        give_access_to_remote_tmpdir = Confirm.ask(
            f'Do you want to give service account {service_account} read/write access to bucket {bucket}?', default=True
        )
        if give_access_to_remote_tmpdir:
            try:
                await grant_service_account_bucket_read_access(fs._storage_client, bucket_info, service_account)
                await grant_service_account_bucket_write_access(fs._storage_client, bucket_info, service_account)
            except Exception as e:
                print(f'ERROR: Could not give service account {service_account} access to bucket {bucket_info.name}. '
                      f'Does the bucket {bucket} exist? '
                      f'Have you installed gcloud? For directions see https://cloud.google.com/sdk/docs/install '
                      f'Do you have admin privileges to grant permissions to resources? '
                      f'Are you logged into your google account by running `gcloud auth login`?')
                print(e)
                errors = True
            else:
                print(f'Granted service account {service_account} read and write access to {bucket}.')

    return (remote_tmpdir, errors)


async def setup_new_remote_tmpdir(bucket_name: str, region: str, project: str, service_account: str) -> Tuple[Optional[str], bool]:
    from hailtop.aiogoogle import GoogleStorageClient  # pylint: disable=import-outside-toplevel
    from .utils import (
        BucketInfo,
        create_gcp_bucket,
        grant_service_account_bucket_read_access,
        grant_service_account_bucket_write_access,
    )  # pylint: disable=import-outside-toplevel

    remote_tmpdir = f'{bucket_name}/batch/tmp'
    errors = False

    retention_days = IntPrompt.ask(f'How many days should files be retained in bucket {bucket_name}?', default=30)
    if retention_days <= 0:
        print(f'invalid value for retention policy in days {retention_days}')
        return (None, True)

    bucket_info = BucketInfo(bucket_name, region, 'region', project, retention_days)

    async with GoogleStorageClient() as storage_client:
        try:
            await create_gcp_bucket(storage_client, bucket_info)
            print(f'Created bucket {bucket_name} in project {project} with retention policy set to {retention_days} days.')
        except Exception as e:
            print(f'ERROR: Could not create bucket {bucket_name}. '
                  f'Do you have admin privileges to create new resources in project {project}? '
                  f'Have you installed gcloud? For directions see https://cloud.google.com/sdk/docs/install '
                  f'Are you logged into your google account by running `gcloud auth login`?')
            print(e)
            errors = True
        else:
            try:
                await grant_service_account_bucket_read_access(storage_client, bucket_info, service_account)
                await grant_service_account_bucket_write_access(storage_client, bucket_info, service_account)
            except Exception as e:
                print(f'ERROR: Could not give service account {service_account} access to bucket {bucket_name}. '
                      f'Do you have admin privileges to grant permissions to resources in project {project}? '
                      f'Have you installed gcloud? For directions see https://cloud.google.com/sdk/docs/install '
                      f'Are you logged into your google account by running `gcloud auth login`?')
                print(e)
                errors = True
            else:
                print(f'Granted service account {service_account} read and write access to {bucket_name}.')

    return (remote_tmpdir, errors)


async def initialize_gcp(username: str, hail_identity: str, supported_regions: List[str], default_region: str) -> Tuple[Optional[str], bool, str]:
    from hailtop.utils import secret_alnum_string  # pylint: disable=import-outside-toplevel

    from .utils import get_gcp_default_project, get_default_region  # pylint: disable=import-outside-toplevel

    default_project = await get_gcp_default_project()
    project = Prompt.ask('Which google project should resources be created in?', default=default_project)

    region = Prompt.ask('Which region should resources be created in?', default=default_region, choices=supported_regions)

    token = secret_alnum_string(5).lower()
    maybe_bucket_name = f'hail-batch-{username}-{token}'

    create_remote_tmpdir = Confirm.ask(f'Do you want to create a new bucket "{maybe_bucket_name}" in project "{project}"?',
                                       default=True)
    if create_remote_tmpdir:
        remote_tmpdir, errors = await setup_new_remote_tmpdir(maybe_bucket_name, region, project, hail_identity)
    else:
        remote_tmpdir, errors = await setup_existing_remote_tmpdir(region, hail_identity)

    return (remote_tmpdir, errors, region)


async def async_basic_initialize(incremental: bool = False):
    from hailtop.auth.auth import async_get_userinfo  # pylint: disable=import-outside-toplevel
    from hailtop.batch_client.aioclient import BatchClient  # pylint: disable=import-outside-toplevel
    from hailtop.config import DeployConfig  # pylint: disable=import-outside-toplevel
    from hailtop.config import get_user_config, get_user_config_path  # pylint: disable=import-outside-toplevel

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
            remote_tmpdir, errors, region = await initialize_gcp(username, hail_identity, supported_regions, default_region)
        else:
            region = Prompt.ask('Which region should resources be created in?', default=default_region,
                                choices=supported_regions)

            remote_tmpdir = Prompt.ask(f'Enter a path to an existing remote temporary directory (ex: gs://my-bucket/batch/tmp)')
            print(f'WARNING: You will need to grant read/write access to {remote_tmpdir} for account {hail_identity}')
            errors = True

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

    get_user_config(reload=True)
    print('FINAL CONFIGURATION:')
    list_config()

    if errors:
        print('Initialized Hail with errors.')
        raise Exit()


def initialize():
    asyncio.get_event_loop().run_until_complete(async_basic_initialize())
