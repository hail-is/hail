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
        if project:
            return project
        return None
    except Exception:
        return None


async def get_regions_with_default(batch_client, cloud: str) -> Tuple[List[str], Optional[str]]:
    supported_regions = await batch_client.supported_regions()

    if cloud == 'gcp':
        if 'us-central1' in supported_regions:
            default = 'us-central1'
        else:
            default = None
    elif cloud == 'azure':
        if 'eastus' in supported_regions:
            default = 'eastus'
        else:
            default = None
    else:
        default = None

    return (supported_regions, default)


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


async def create_gcp_bucket(storage_client, bucket: str):
    from hailtop.utils import check_shell  # pylint: disable=import-outside-toplevel

    {
        "rule":
            [
                {
                    "action": {"type": "Delete"},
                    "condition": {"age": 365}
                }
            ]
    }
    await check_shell(f"gcloud storage buckets add-iam-policy-binding gs://{bucket} \
    --member={service_account_member} \
    --role=roles/storage.objectViewer")

    body = {
        'location': bucket_info.location,
        'locationType': bucket_info.location_type,
        'lifecycle': {'rule': [{'action': {'type': 'Delete', 'condition': {'age': bucket_info.retention_policy_days}}}]}
    }
    await storage_client.insert_bucket(bucket_info.name, bucket_info.project, body=body)


async def grant_service_account_bucket_read_access(bucket: str, service_account: str):
    from hailtop.utils import check_shell  # pylint: disable=import-outside-toplevel
    service_account_member = f'serviceAccount:{service_account}'
    await check_shell(f"gcloud storage buckets add-iam-policy-binding gs://{bucket} \
    --member={service_account_member} \
    --role=roles/storage.objectViewer", echo=True)


async def grant_service_account_bucket_write_access(bucket: str, service_account: str):
    from hailtop.utils import check_shell  # pylint: disable=import-outside-toplevel
    service_account_member = f'serviceAccount:{service_account}'
    await check_shell(f"gcloud storage buckets add-iam-policy-binding gs://{bucket} \
    --member={service_account_member} \
    --role=roles/storage.objectCreator", echo=True)
