import asyncio
import json
import os
import typer
from typing_extensions import Annotated as Ann
from typing import List, Optional, Tuple

from typer import Abort, Option as Opt
from rich.prompt import Confirm, IntPrompt, Prompt

from hailtop.batch_client.aioclient import BatchClient
from hailtop.fs import AsyncFS
from hailtop.aiogoogle import GoogleStorageClient
from hailtop.utils import CalledProcessError, check_shell, check_shell_output
from hailtop.fs.router_fs import RouterAsyncFS

from .utils import *


def get_domain_namespace() -> (str, str):
    domain = Prompt.ask('What domain is the Hail service running in?', default='hail.is')
    namespace = Prompt.ask('What namespace is the Hail service running in?', default='default')
    return (domain, namespace)


async def get_cloud(client: BatchClient) -> str:
    cloud = await client.cloud()
    return cloud


# async def get_gcp_default_project() -> str:
#
#     try:
#         project, _ = await check_shell_output("gcloud config list --format 'value(core.project)'")
#         project = project.strip().decode('utf-8')
#         if project:
#             use_default_project = Confirm(f'Found default GCP project is {project}. Do you want to use this project?')
#             if not use_default_project:
#                 project = Prompt.ask('Enter GCP project to use')
#         else:
#             project = Prompt.ask('Enter GCP project to use')
#     except CalledProcessError:
#         project = Prompt.ask('Enter GCP project to use')
#
#     print(f'Using GCP project {project}')
#     return project


def get_gcp_requester_pays_project(default_gcp_project: str) -> Optional[str]:
    set_project = Confirm.ask('Do you want to set a project that will be billed when accessing requester pays buckets?',
                              default=False)
    if not set_project:
        return None

    requester_pays_project = Prompt.ask('Enter a GCP project to use for requester pays buckets',
                                        default=default_gcp_project)
    return requester_pays_project


def get_gcp_requester_pays_allowed_buckets() -> Optional[List[str]]:
    set_buckets = Confirm.ask('Do you want to set allowed buckets when using requester pays in GCS?', default=False)
    if not set_buckets:
        return None

    selected_buckets = []
    while True:
        bucket = Prompt.ask('Enter a bucket to allow when using requester pays in GCS')
        if bucket is None:
            break
        selected_buckets.append(bucket)

    return selected_buckets


async def get_or_create_gcp_remote_tmpdir(
        storage_client: GoogleStorageClient,
        remote_tmpdir: str,
        project: str,
        bucket: str,
        hail_identity: str,
        regions: List[str],
) -> Tuple[str, str]:
    try:
        bucket_info = await storage_client.get_bucket(bucket)
        bucket_region = bucket_info['location']
        bucket_location_type = bucket_info['locationType']
    except aiohttp.ClientResponseError as e:
        if e.status == 404:
            bucket_info = None
        raise

    bucket_exists = bucket_info is not None
    if not bucket_exists:
        make_bucket = Prompt.ask(f'Would you like to create bucket {bucket} in project {project}?')
        if make_bucket:
            retention_policy_days = IntPrompt.ask('How many days would you like objects to live for before being automatically deleted?', default='30')
            bucket_region = Prompt.ask(f'Which region should the bucket {bucket} be created in?', choices=regions)
            bucket_location_type = 'region'
            body = {
                'location': bucket_region,
                'locationType': bucket_location_type,
                'lifecycle': {'rule': [{'action': {'type': 'Delete', 'condition': {'age': retention_policy_days}}}]}
            }
            await storage_client.insert_bucket(bucket, project)
            print(f'Created bucket {bucket} in project {project}.')
        else:
            print(f'WARNING: remote_tmpdir location {remote_tmpdir} does not exist! Do you have access to this bucket if it exists?')
            return (None, None)

    service_account_member = f'serviceAccount:{hail_identity}'
    has_write_access = False
    has_read_access = False

    write_roles = (
        'roles/storage.legacyBucketOwner',
        'roles/storage.legacyBucketWriter',
        'roles/storage.objectAdmin',
        'roles/storage.objectCreator',
    )
    read_roles = (
        'roles/storage.legacyBucketOwner',
        'roles/storage.legacyBucketReader',
        'roles/storage.objectAdmin',
        'roles/storage.objectViewer',
    )

    iam_policy = await storage_client.get_bucket_iam_policy(bucket)
    bindings = iam_policy['bindings']
    for binding in bindings:
        role = binding['role']
        if role in write_roles:
            if service_account_member in binding['members']:
                has_write_access = True
        if role in read_roles:
            if service_account_member in binding['members']:
                has_read_access = True

    if not has_read_access:
        give_read_access = Prompt.ask(
            f'Would you like to give service account {hail_identity} read access to bucket {bucket}?')
        if give_read_access:
            await storage_client.grant_bucket_read_access(bucket, service_account_member)
            print(f'Granted service account {hail_identity} read access to bucket {bucket} in project {project}.')
        else:
            print(
                f'WARNING: service account {hail_identity} does not have read access to the remote tmpdir {remote_tmpdir}!')

    if not has_write_access:
        give_write_access = Prompt.ask(f'Would you like to give service account {hail_identity} write access to bucket {bucket}?')
        if give_write_access:
            await storage_client.grant_bucket_write_access(bucket, service_account_member)
            print(f'Granted service account {hail_identity} write access to bucket {bucket} in project {project}.')
        else:
            print(f'WARNING: service account {hail_identity} does not have write access to the remote tmpdir {remote_tmpdir}!')

    return (bucket_region, bucket_location_type)


async def get_remote_tmpdir_location(router_fs: RouterAsyncFS, cloud: str, hail_identity: str) -> Tuple[str, AsyncFS]:
    from hailtop.aiocloud import aiogoogle, aioazure
    while True:
        remote_tmpdir_location = Prompt.ask('Enter a remote temporary directory')
        fs = router_fs._get_fs(remote_tmpdir_location)
        if ((cloud == 'gcp' and not isinstance(fs, aiogoogle.GoogleStorageAsyncFS)) or
                (cloud == 'azure' and not isinstance(fs, aioazure.AzureAsyncFS))):
            print(f'Invalid remote temporary directory for cloud {cloud}.')
        return (remote_tmpdir_location, fs)


async def get_regions(client: BatchClient, cloud: str) -> Optional[List[str]]:
    all_regions = 'ALL'

    supported_regions = await client.supported_regions()

    if cloud == 'gcp':
        if 'us-central1' in supported_regions:
            default = 'us-central1'
        else:
            default = all_regions
    elif cloud == 'azure':
        if 'eastus' in supported_regions:
            default = 'eastus'
        else:
            default = all_regions
    else:
        default = all_regions

    select_any_region = Confirm.ask('Do you want to specify which regions a job runs in?', default=True)
    if not select_any_region:
        return None

    select_all_regions = Confirm.ask('Do you want jobs to be able to run in all supported regions?', default=False)
    if select_all_regions:
        print(f'Selected regions: {supported_regions}')
        return supported_regions

    selected_regions = []
    while True:
        choices = list(set(supported_regions) - set(selected_regions))
        if default is not None and default not in choices:
            default = None
        region = Prompt.ask('Which region should jobs run in?', default=default, choices=choices)
        if region is None:
            break
        selected_regions.append(region)

    print(f'Selected regions: {selected_regions}')
    return selected_regions


# async def check_for_gcloud() -> bool:
#     try:
#         await check_shell("gcloud version")
#         return True
#     except Exception:
#         return False


async def check_artifact_registry_existence(gcp_project: str) -> bool:
    try:
        await check_shell(f'gcloud --project {gcp_project} artifacts repositories list')
        return True
    except CalledProcessError as e:
        print(f'Error listing repositories in the artifact registry for project {gcp_project}.')
        print('This error could be due to the artifact registry not being enabled in your project.')
        print(f'To enable the artifact registry run the following command: `gcloud --project {gcp_project} services enable artifactregistry.googleapis.com`')
        print('Note: you need sufficient privileges in your project to run this operation.')
        print(e.stdout)
        print(e.stderr)
        return False


async def create_docker_repository(project: str, username: str, default_region: str, service_account: str):
    default_repo_name = f'{username}-batch'
    create_repo = Prompt.ask('Do you want to create a container repository in the artifact registry?', default=True)
    if create_repo:
        repo_name = Prompt.ask('What is the name of your repository?', default=default_repo_name)
        default_multi_region_location = default_region.split('-')[0]
        location = Prompt.ask('What location should your repository be created in?', default=default_multi_region_location)
        if location != default_multi_region_location and location != default_region:
            print(f'The selected location {location} is not located within the same region as Batch jobs configured to run in {default_region}.')
            proceed = Prompt.ask('Would you like to proceed anyways?', default=False)
            if not proceed:
                print(f'Skipped creating repository {repo_name} in the artifact registry for project {project}.')
                return
        try:
            await check_shell(
                f'''
gcloud artifacts repositories create {repo_name} \
    --repository-format=docker \
    --location={location} \
    --description="Docker registry for using Hail Batch" \
    --immutable-tags
''')
        except CalledProcessError as e:
            print(f'Error while creating repository {repo_name} in the artifact registry for project {project}.')
            print('Skipping repository creation.')
            print(f'stdout: {e.stdout}')
            print(f'stderr: {e.stderr}')
            return

        try:
            await check_shell(
                f'''
gcloud artifacts repositories add-iam-policy-binding {repo_name} \
   --location {location} \
   --member={service_account} \
   --role=roles/artifactregistry.writer
''')
        except CalledProcessError as e:
            print(f'Error while granting permissions to {repo_name} for service account {service_account}.')
            print('Skipping granting permissions.')
            print(f'stdout: {e.stdout}')
            print(f'stderr: {e.stderr}')
            return


async def basic_initialize(overwrite: bool):
    from hailtop.aiocloud.aiogoogle import GoogleStorageAsyncFS  # pylint: disable=import-outside-toplevel
    from hailtop.auth.auth import async_get_userinfo  # pylint: disable=import-outside-toplevel
    from hailtop.config import DeployConfig  # pylint: disable=import-outside-toplevel
    from hailtop.config import get_user_config_path  # pylint: disable=import-outside-toplevel

    from ..auth.login import async_login  # pylint: disable=import-outside-toplevel
    from ..config.cli import set as set_config, list as list_config

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
        if cloud == 'gcp':
            gcloud_installed = await check_for_gcloud()
            if not gcloud_installed:
                print('gcloud is not installed or on the path. Install gcloud before retrying `hailctl init`.\n'
                      'For directions see https://cloud.google.com/sdk/docs/install')
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

                bucket_info = BucketInfo(maybe_bucket_name, region, 'region', project, retention_days)

                async with GoogleStorageClient() as storage_client:
                    await create_gcp_bucket(storage_client, bucket_info)
                    await grant_service_account_bucket_read_access(storage_client, bucket_info, hail_identity)
                    await grant_service_account_bucket_write_access(storage_client, bucket_info, hail_identity)

                print(f'Created bucket {maybe_bucket_name} in project {project} with retention policy set to {retention_days} days.')
                print(f'Also granted service account {hail_identity} read and write access to this bucket.')

                remote_tmpdir = f'{maybe_bucket_name}/batch/tmp'
            else:
                remote_tmpdir = Prompt.ask(f'Enter a path to an existing remote temporary directory (ex: gs://my-bucket/batch/tmp)')

                async with GoogleStorageAsyncFS() as fs:
                    bucket, _ = fs.get_bucket_and_name(remote_tmpdir)
                    bucket_info = await get_gcp_bucket_information(fs._storage_client, bucket)
                    if bucket_info.location != region or bucket_info.is_multi_regional():
                        print(f'WARNING: given remote temporary directory {remote_tmpdir} is not located in {region} or is multi-regional.')

    config_file = get_user_config_path()

    if os.path.isfile(config_file):
        os.remove(config_file)

    set_config('domain', domain)
    set_config('batch/billing_project', trial_bp_name)
    set_config('batch/remote_tmpdir', remote_tmpdir)
    set_config('batch/regions', region)
    set_config('batch/backend', 'service')
    set_config('query/backend', 'batch')



async def async_initialize():
    from hailtop.config import DeployConfig
    from hailtop.auth.auth import async_get_userinfo, hail_credentials
    from hailtop.hailctl.auth.login import async_login
    from hailtop.fs.router_fs import RouterAsyncFS
    from hailtop.aiocloud import aiogoogle, aioazure
    from hailtop.hailctl.config.cli import set as set_config, list as list_config
    from hailtop.config import get_user_config, get_user_config_path  # pylint: disable=import-outside-toplevel

    domain, namespace = get_domain_namespace()
    deploy_config = DeployConfig('external', namespace, domain)

    config_file = os.environ.get('HAIL_DEPLOY_CONFIG_FILE', os.path.expanduser('~/.hail/deploy-config.json'))
    config = deploy_config.get_config()
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f)

    await async_login(namespace)

    user_info = await async_get_userinfo()
    hail_identity = user_info['hail_identity']

    billing_project = Prompt.ask('Enter a Batch billing project to use', default=user_info['trial_bp_name'])

    client = await BatchClient.create(billing_project, deploy_config=deploy_config)

    try:
        cloud = await get_cloud(client)
        print(f'The service located at {domain} is in {cloud}.')

        regions = await get_regions(client, cloud)

        async with RouterAsyncFS() as router_fs:
            if cloud == 'gcp':
                await check_for_gcloud()
                gcp_project = await get_gcp_default_project()
                requester_pays_project = get_gcp_requester_pays_project(gcp_project)
                requester_pays_buckets = get_gcp_requester_pays_allowed_buckets()

                remote_tmpdir_location, fs = await get_remote_tmpdir_location(cloud, hail_identity)
                assert isinstance(fs, aiogoogle.GoogleStorageAsyncFS)

                bucket, _ = fs.get_bucket_and_name(remote_tmpdir_location)
                bucket_region, bucket_type = await get_or_create_gcp_remote_tmpdir(
                    fs._storage_client,
                    remote_tmpdir,
                    gcp_project,
                    bucket,
                    hail_identity,
                    regions,
                )

                if bucket_type != 'region':
                    print(f'WARNING: remote_tmpdir bucket {bucket} is not a REGIONAL bucket!')
                if bucket_region not in regions:
                    print(f'WARNING: remote_tmpdir bucket {bucket} in not located in a Batch region {regions}!')

                if len(regions) > 1:
                    print(f'WARNING: remote_tmpdir bucket {bucket} is location in {bucket_region}, but jobs can run in multiple regions {regions}!')
                    change_regions = Prompt.ask(f'Do you want to modify which regions jobs run in to only run in the same region as bucket {bucket}?',
                                                default=True)
                    if change_regions:
                        regions = [bucket_region]

                artifact_registry_exists = await check_artifact_registry_existance(gcp_project)
                if artifact_registry_exists:
                    await create_docker_repository(gcp_project, user_info['username'], bucket_region)
            else:
                remote_tmpdir_location, fs = await get_remote_tmpdir_location(cloud, hail_identity)
                # FIXME: Get or make storage accounts etc. in other clouds

        batch_backend = 'service'

        query_backend = Prompt.ask('Backend to use for Hail Query.', choices=['local', 'spark', 'batch'], default='batch')

        query_driver_cores = IntPrompt.ask('Default number of cores to use for Query driver jobs', choices=['1', '2', '4', '8'], default='1')
        query_worker_cores = IntPrompt.ask('Default number of cores to use for Query worker jobs', choices=['1', '2', '4', '8'], default='1')

        config_file = get_user_config_path()

        if os.path.isfile(config_file):
            os.remove(config_file)

        set_config('domain', domain)
        set_config('default_namespace', namespace)
        if requester_pays_project:
            set_config('gcs_requester_pays/project', requester_pays_project)
        if requester_pays_buckets:
            set_config('gcs_requester_pays/buckets', ','.join(requester_pays_buckets))
        set_config('batch/billing_project', billing_project)
        set_config('batch/remote_tmpdir', remote_tmpdir_location)
        set_config('batch/regions', ','.join(regions))
        set_config('batch/backend', batch_backend)
        set_config('query/backend', query_backend)
        set_config('query/batch_driver_cores', query_driver_cores)
        set_config('query/batch_worker_cores', query_worker_cores)

        # FIXME: Add prompts and checks for the following
        # query/batch_driver_memory
        # query/batch_worker_memory
        # query/name_prefix

        print('Finished initializing Hail!')

        get_user_config(reload=True)
        list_config()
    finally:
        await client.close()


def initialize():
    asyncio.get_event_loop().run_until_complete(async_initialize())
