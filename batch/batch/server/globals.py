import os
import datetime
import collections
import asyncio

from google.cloud import storage
from google.oauth2 import service_account
import hailjwt as hj

from .google_storage import upload_private_gs_file_from_string, download_gs_file_as_string
from .google_storage import exists_gs_file


batch_gsa_key = os.environ.get('BATCH_GSA_KEY', '/batch-gsa-key/privateKeyData')
credentials = service_account.Credentials.from_service_account_file(batch_gsa_key)
gcs_client = storage.Client(credentials=credentials)

batch_jwt = os.environ.get('BATCH_JWT', '/batch-jwt/jwt')
with open(batch_jwt, 'r') as f:
    batch_bucket_name = hj.JWTClient.unsafe_decode(f.read())['bucket_name']


pod_name_job = {}
job_id_job = {}
batch_id_batch = {}


def _gs_log_path(instance_id, job_id, task_name):
    return f'{instance_id}/{job_id}/{task_name}/job.log'


async def write_gs_log_file(thread_pool, instance_id, job_id, task_name, log):
    path = _gs_log_path(instance_id, job_id, task_name)
    await blocking_to_async(thread_pool, upload_private_gs_file_from_string, gcs_client, batch_bucket_name, path, log)
    return path


async def read_gs_log_file(thread_pool, instance_id, job_id, task_name):
    path = _gs_log_path(instance_id, job_id, task_name)
    if await blocking_to_async(thread_pool, exists_gs_file, gcs_client, batch_bucket_name, path):
        return await blocking_to_async(thread_pool, download_gs_file_as_string, gcs_client, batch_bucket_name, path)
    return None


_counter = 0


def max_id():
    return _counter


def next_id():
    global _counter

    _counter = _counter + 1
    return _counter


_recent_events = collections.deque(maxlen=50)


def get_recent_events():
    return _recent_events


def add_event(event):
    global _recent_events

    event['time'] = str(datetime.datetime.now())
    _recent_events.append(event)


async def blocking_to_async(thread_pool, f, *args, **kwargs):
    return await asyncio.get_event_loop().run_in_executor(
        thread_pool, lambda: f(*args, **kwargs))
