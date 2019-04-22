from google.cloud import storage

from .globals import blocking_to_async


gcs_client = storage.Client()


async def upload_private_gs_file_from_string(thread_pool, bucket, target_path, string):
    bucket = gcs_client.bucket(bucket)
    f = bucket.blob(target_path)
    f.metadata = {'Cache-Control': 'no-cache'}
    await blocking_to_async(thread_pool, f.upload_from_string(string))


async def download_gs_file_as_string(thread_pool, bucket, path):
    bucket = gcs_client.bucket(bucket)
    f = bucket.blob(path)
    f.metadata = {'Cache-Control': 'no-cache'}
    content = await blocking_to_async(thread_pool, f.download_as_string())
    return content.decode('utf-8')


def exists_gs_file(bucket, path):
    bucket = gcs_client.bucket(bucket)
    f = bucket.blob(path)
    return f.exists()
