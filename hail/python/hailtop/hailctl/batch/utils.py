import json
import tempfile
from typing import Dict, Optional

from hailtop.config import ConfigVariable


class InsufficientPermissions(Exception):
    def __init__(self, message: str):
        self.message = message


async def already_logged_into_service() -> bool:
    from hailtop.auth.auth import async_get_userinfo  # pylint: disable=import-outside-toplevel
    try:
        await async_get_userinfo()
        return True
    except Exception:
        return False


async def login_to_service(domain: Optional[str]):
    from hailtop.hailctl.auth.login import async_login  # pylint: disable=import-outside-toplevel
    from hailtop.hailctl.config.cli import set as _set  # pylint: disable=import-outside-toplevel
    if domain:
        _set(ConfigVariable.DOMAIN, domain)
    await async_login('default')


async def check_for_gcloud() -> bool:
    from hailtop.utils import check_exec_output  # pylint: disable=import-outside-toplevel
    try:
        await check_exec_output('gcloud', 'version')
        return True
    except Exception:
        return False


async def get_gcp_default_project(verbose: bool) -> Optional[str]:
    from hailtop.utils import check_exec_output  # pylint: disable=import-outside-toplevel
    try:
        project, _ = await check_exec_output('gcloud', 'config', 'get-value', 'project', echo=verbose)
        project = project.strip().decode('utf-8')
        return project
    except Exception:
        return None


async def get_gcp_bucket_information(bucket: str, verbose: bool) -> Optional[dict]:
    from hailtop.utils import CalledProcessError, check_exec_output  # pylint: disable=import-outside-toplevel
    try:
        info, _ = await check_exec_output('gcloud', 'storage', 'buckets', 'describe', f'gs://{bucket}', '--format="json"', echo=verbose)
        return json.loads(info.decode('utf-8'))
    except CalledProcessError as e:
        if 'does not have storage.buckets.get access to the Google Cloud Storage bucket' in e.stderr.decode('utf-8'):
            msg = f'ERROR: You do not have sufficient permissions to get information about bucket {bucket} or it does not exist. ' \
                  f'If the bucket exists, ask a project administrator to give you the permission "storage.buckets.get" or ' \
                  f'assign you the StorageAdmin role in Google Cloud Storage.'
            raise InsufficientPermissions(msg) from e
        raise


async def create_gcp_bucket(*,
                            project: str,
                            bucket: str,
                            location: str,
                            lifecycle_days: Optional[int],
                            labels: Optional[Dict[str, str]],
                            verbose: bool):
    from hailtop.utils import CalledProcessError, check_exec_output  # pylint: disable=import-outside-toplevel
    if labels:
        labels_str = ','.join(f'{k}={v}' for k, v in labels.items())
    else:
        labels_str = None

    try:
        await check_exec_output('gcloud', '--project', project, 'storage', 'buckets', 'create', f'gs://{bucket}', f'--location={location}', echo=verbose)
    except CalledProcessError as e:
        if 'does not have storage.buckets.create access to the Google Cloud project' in e.stderr.decode('utf-8'):
            msg = f'ERROR: You do not have the necessary permissions to create buckets in project {project}. Ask a project administrator ' \
                  f'to give you the permission "storage.buckets.create" or assign you the StorageAdmin role or ask them to create the bucket {bucket} on your behalf.'
            raise InsufficientPermissions(msg) from e
        raise

    try:
        if lifecycle_days:
            lifecycle_policy = {
                "rule": [
                    {
                        "action": {"type": "Delete"},
                        "condition": {"age": lifecycle_days}
                    }
                ]
            }

            with tempfile.NamedTemporaryFile(mode='w') as f:
                f.write(json.dumps(lifecycle_policy))
                f.flush()
                await check_exec_output('gcloud', '--project', project, 'storage', 'buckets', 'update', f'--lifecycle-file="{f.name}"', f'gs://{bucket}', echo=verbose)

        if labels_str:
            await check_exec_output('gcloud', '--project', project, 'storage', 'buckets', 'update', f'--update-labels={labels_str}', f'gs://{bucket}', echo=verbose)
    except CalledProcessError as e:
        if 'does not have storage.buckets.get access to the Google Cloud Storage bucket' in e.stderr.decode('utf-8'):
            msg = f'ERROR: You do not have the necessary permissions to update bucket {bucket} in project {project}. Ask a project administrator ' \
                  f'to assign you the StorageAdmin role in Google Cloud Storage for bucket {bucket} or ask them to update the bucket {bucket} on your behalf.'
            if lifecycle_days:
                msg += f'Update the bucket to have a lifecycle policy of {lifecycle_days} days.'
            if labels_str:
                msg += f'Update the bucket to have labels: {labels_str}'
            raise InsufficientPermissions(msg) from e
        raise


async def grant_service_account_bucket_access_with_role(project: Optional[str], bucket: str, service_account: str, role: str, verbose: bool):
    from hailtop.utils import CalledProcessError, check_exec_output  # pylint: disable=import-outside-toplevel

    if project:
        project = f'--project {project}'
    else:
        project = ''
    try:
        service_account_member = f'serviceAccount:{service_account}'
        await check_exec_output('gcloud', project, 'storage', 'buckets', 'add-iam-policy-binding', f'gs://{bucket}', '--member', service_account_member, '--role', role,
                                echo=verbose)
    except CalledProcessError as e:
        if 'does not have storage.buckets.getIamPolicy access to the Google Cloud Storage bucket' in e.stderr.decode('utf-8'):
            msg = f'ERROR: You do not have the necessary permissions to set permissions for bucket {bucket} in project {project}. Ask a project administrator ' \
                  f'to assign you the StorageIAMAdmin role in Google Cloud Storage or ask them to update the permissions on your behalf by giving ' \
                  f'service account {service_account} the role "{role}" for bucket {bucket}.'
            raise InsufficientPermissions(msg) from e
        raise
