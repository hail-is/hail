import os
import asyncio

from google.cloud import storage
from google.oauth2 import service_account
import google.api_core
import hailjwt as hj

from .google_storage import upload_private_gs_file_from_string, download_gs_file_as_string
from .google_storage import delete_gs_file


batch_gsa_key = os.environ.get('BATCH_GSA_KEY', '/batch-gsa-key/privateKeyData')
credentials = service_account.Credentials.from_service_account_file(batch_gsa_key)
gcs_client = storage.Client(credentials=credentials)

batch_jwt = os.environ.get('BATCH_JWT', '/batch-jwt/jwt')
with open(batch_jwt, 'r') as f:
    batch_bucket_name = hj.JWTClient.unsafe_decode(f.read())['bucket_name']


def _gs_log_path(instance_id, job_id, task_name):
    return f'{instance_id}/{job_id}/{task_name}/job.log'


async def write_gs_log_file(thread_pool, instance_id, job_id, task_name, log):
    path = _gs_log_path(instance_id, job_id, task_name)
    await blocking_to_async(thread_pool, upload_private_gs_file_from_string, gcs_client, batch_bucket_name, path, log)
    return f'gs://{batch_bucket_name}/{path}'


async def read_gs_log_file(thread_pool, uri):
    if uri is not None:
        assert uri.startswith('gs://')
        uri = uri.lstrip('gs://').split('/')
        bucket_name = uri[0]
        path = '/'.join(uri[1:])
        try:
            return await blocking_to_async(thread_pool, download_gs_file_as_string, gcs_client, bucket_name, path)
        except google.api_core.exceptions.NotFound:
            return None
    return None


async def delete_gs_log_file(thread_pool, instance_id, job_id, task_name):
    path = _gs_log_path(instance_id, job_id, task_name)
    await blocking_to_async(thread_pool, delete_gs_file, gcs_client, batch_bucket_name, path)


async def blocking_to_async(thread_pool, f, *args, **kwargs):
    return await asyncio.get_event_loop().run_in_executor(
        thread_pool, lambda: f(*args, **kwargs))
