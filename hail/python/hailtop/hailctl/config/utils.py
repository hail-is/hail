import aiohttp

from typing import List, Optional, Tuple


async def already_logged_into_service() -> bool:
    from hailtop.auth.auth import async_get_userinfo  # pylint: disable=import-outside-toplevel
    try:
        await async_get_userinfo()
        return True
    except Exception:
        return False


async def check_for_gcloud() -> bool:
    from hailtop.utils import check_shell  # pylint: disable=import-outside-toplevel
    try:
        await check_shell("gcloud version")
        return True
    except Exception:
        return False


async def get_gcp_default_project() -> Optional[str]:
    from hailtop.utils import check_shell_output  # pylint: disable=import-outside-toplevel
    try:
        project, _ = await check_shell_output("gcloud config list --format 'value(core.project)'")
        project = project.strip().decode('utf-8')
        return project
    except Exception:
        return None


def get_default_region(supported_regions: List[str], cloud: str) -> Optional[str]:
    if cloud == 'gcp':
        if 'us-central1' in supported_regions:
            return 'us-central1'
    elif cloud == 'azure':
        if 'eastus' in supported_regions:
            return 'eastus'

    return None


class BucketInfo:
    def __init__(self,
                 name: str,
                 location: str,
                 location_type: str,
                 project: Optional[str],
                 retention_policy_days: Optional[int] = None):
        self.name = name
        self.location = location
        self.location_type = location_type
        self.project = project
        self.retention_policy_days = retention_policy_days

    def is_multi_regional(self):
        return self.location_type.lower() != 'region'


async def get_gcp_bucket_information(storage_client, bucket: str) -> Optional[BucketInfo]:
    try:
        bucket_info = await storage_client.get_bucket(bucket)
        bucket_region = bucket_info['location'].lower()
        bucket_location_type = bucket_info['locationType'].lower()
        project = None
        return BucketInfo(bucket, bucket_region, bucket_location_type, project)
    except aiohttp.ClientResponseError as e:
        if e.status == 404:
            return None
        raise


async def create_gcp_bucket(storage_client, bucket_info: BucketInfo):
    body = {
        'name': bucket_info.name,
        'location': bucket_info.location,
        'locationType': bucket_info.location_type,
        'lifecycle': {'rule': [{'action': {'type': 'Delete'}, 'condition': {'age': bucket_info.retention_policy_days}}]}
    }
    await storage_client.insert_bucket(bucket_info.project, body=body)


async def check_service_account_has_bucket_read_access(storage_client, bucket_info: BucketInfo, service_account: str) -> bool:
    service_account_member = f'serviceAccount:{service_account}'
    read_roles = (
        'roles/storage.legacyBucketOwner',
        'roles/storage.legacyBucketReader',
        'roles/storage.objectAdmin',
        'roles/storage.objectViewer',
    )

    iam_policy = await storage_client.get_bucket_iam_policy(bucket_info.name)
    bindings = iam_policy['bindings']
    for binding in bindings:
        role = binding['role']
        if role in read_roles:
            if service_account_member in binding['members']:
                return True

    return False


async def grant_service_account_bucket_read_access(storage_client, bucket_info: BucketInfo, service_account: str):
    service_account_member = f'serviceAccount:{service_account}'
    await storage_client.grant_bucket_read_access(bucket_info.name, service_account_member)


async def check_service_account_has_bucket_write_access(storage_client, bucket_info: BucketInfo, service_account: str) -> bool:
    service_account_member = f'serviceAccount:{service_account}'
    write_roles = (
        'roles/storage.legacyBucketOwner',
        'roles/storage.legacyBucketWriter',
        'roles/storage.objectAdmin',
        'roles/storage.objectCreator',
    )

    iam_policy = await storage_client.get_bucket_iam_policy(bucket_info.name)
    bindings = iam_policy['bindings']
    for binding in bindings:
        role = binding['role']
        if role in write_roles:
            if service_account_member in binding['members']:
                return True

    return False


async def grant_service_account_bucket_write_access(storage_client, bucket_info: BucketInfo, service_account: str):
    service_account_member = f'serviceAccount:{service_account}'
    await storage_client.grant_bucket_write_access(bucket_info.name, service_account_member)
