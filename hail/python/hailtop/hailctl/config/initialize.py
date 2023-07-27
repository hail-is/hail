import asyncio
import os
import typer

from typer import Abort, Option as Opt
from rich.prompt import Confirm, IntPrompt, Prompt


async def async_basic_initialize(incremental: bool = False):
    from hailtop.aiogoogle import GoogleStorageAsyncFS, GoogleStorageClient  # pylint: disable=import-outside-toplevel
    from hailtop.auth.auth import async_get_userinfo  # pylint: disable=import-outside-toplevel
    from hailtop.batch_client.aioclient import BatchClient  # pylint: disable=import-outside-toplevel
    from hailtop.config import DeployConfig  # pylint: disable=import-outside-toplevel
    from hailtop.config import get_user_config, get_user_config_path  # pylint: disable=import-outside-toplevel

    from ..auth.login import async_login  # pylint: disable=import-outside-toplevel
    from .cli import set as set_config, list as list_config  # pylint: disable=import-outside-toplevel
    from .utils import (
        BucketInfo,
        already_logged_into_service,
        check_for_gcloud,
        create_gcp_bucket,
        get_gcp_bucket_information,
        get_gcp_default_project,
        get_regions_with_default,
        grant_service_account_bucket_read_access,
        grant_service_account_bucket_write_access,
    )  # pylint: disable=import-outside-toplevel

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

    errors = False

    async with batch_client:
        cloud = await batch_client.cloud()
        if cloud == 'gcp':
            gcloud_installed = await check_for_gcloud()
            if not gcloud_installed:
                print('gcloud is not installed or on the path. Install gcloud before retrying `hailctl init`.\n'
                      'For directions see https://cloud.google.com/sdk/docs/install')
                errors = True
                Abort()

            default_project = await get_gcp_default_project()
            project = Prompt.ask('Which google project should resources be created in?', default=default_project)

            regions, default_region = get_regions_with_default(batch_client, cloud)
            region = Prompt.ask('Which region should resources be created in?', default=default_region, choices=regions)

            maybe_bucket_name = f'hail-batch-{username}'

            create_remote_tmpdir = Prompt.ask(f'Do you want to create a new bucket "{maybe_bucket_name}" in project "{project}"?', default=True)
            if create_remote_tmpdir:
                retention_days = IntPrompt.ask(f'How many days should files be retained in bucket {maybe_bucket_name}?', default=30)
                if retention_days <= 0:
                    print(f'invalid value for retention policy in days {retention_days}')
                    errors = True
                    Abort()

                bucket_info = BucketInfo(maybe_bucket_name, region, 'region', project, retention_days)

                async with GoogleStorageClient() as storage_client:
                    try:
                        await create_gcp_bucket(storage_client, bucket_info)
                        remote_tmpdir = f'{maybe_bucket_name}/batch/tmp'
                        print(f'Created bucket {maybe_bucket_name} in project {project} with retention policy set to {retention_days} days.')
                    except Exception:
                        print(f'ERROR: Could not create bucket {maybe_bucket_name}. '
                              f'Do you have admin privileges to create new resources in project {project}? '
                              f'Are you logged into your google account by running `gcloud auth login`?')
                        remote_tmpdir = None
                        errors = True
                    else:
                        try:
                            await grant_service_account_bucket_read_access(storage_client, bucket_info, hail_identity)
                            await grant_service_account_bucket_write_access(storage_client, bucket_info, hail_identity)
                        except Exception:
                            print(f'ERROR: Could not give service account {hail_identity} access to bucket {maybe_bucket_name}. '
                                  f'Do you have admin privileges to grant permissions to resources in project {project}? '
                                  f'Are you logged into your google account by running `gcloud auth login`?')
                            errors = True
                        else:
                            print(f'Granted service account {hail_identity} read and write access to {maybe_bucket_name}.')
            else:
                remote_tmpdir = Prompt.ask(f'Enter a path to an existing remote temporary directory (ex: gs://my-bucket/batch/tmp)')

                async with GoogleStorageAsyncFS() as fs:
                    bucket, _ = fs.get_bucket_and_name(remote_tmpdir)
                    bucket_info = await get_gcp_bucket_information(fs._storage_client, bucket)
                    if bucket_info.location != region or bucket_info.is_multi_regional():
                        print(f'WARNING: given remote temporary directory {remote_tmpdir} is not located in {region} or is multi-regional.')
                        errors = True

                    give_access_to_remote_tmpdir = Prompt.ask(f'Do you want to give service account {hail_identity} read/write access to bucket {maybe_bucket_name}?', default=True)
                    if give_access_to_remote_tmpdir:
                        try:
                            await grant_service_account_bucket_read_access(fs._storage_client, bucket_info, hail_identity)
                            await grant_service_account_bucket_write_access(fs._storage_client, bucket_info, hail_identity)
                        except Exception:
                            print(f'ERROR: Could not give service account {hail_identity} access to bucket {bucket_info.name}. '
                                  f'Do you have admin privileges to grant permissions to resources? '
                                  f'Are you logged into your google account by running `gcloud auth login`?')
                            errors = True
                        else:
                            print(f'Granted service account {hail_identity} read and write access to {maybe_bucket_name}.')

    Abort()

    config_file = get_user_config_path()

    if not incremental:
        if os.path.isfile(config_file):
            os.remove(config_file)

    set_config('domain', domain)
    set_config('batch/billing_project', trial_bp_name)
    if remote_tmpdir:
        set_config('batch/remote_tmpdir', remote_tmpdir)
    set_config('batch/regions', region)
    set_config('batch/backend', 'service')
    set_config('query/backend', 'batch')

    get_user_config(reload=True)
    list_config()

    if errors:
        print('Initialized Hail with errors.')
        Abort()


def initialize():
    asyncio.get_event_loop().run_until_complete(async_basic_initialize())
