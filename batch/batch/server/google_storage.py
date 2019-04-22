from google.cloud import storage



gcs_client = storage.Client()


async def upload_private_gs_file_from_string(bucket, target_path, string):
    from .server import blocking_to_async
    bucket = gcs_client.bucket(bucket)
    f = bucket.blob(target_path)
    f.metadata = {'Cache-Control': 'no-cache'}
    await blocking_to_async(f.upload_from_string(string))


async def download_gs_file_as_string(bucket, path):
    from .server import blocking_to_async
    bucket = gcs_client.bucket(bucket)
    f = bucket.blob(path)
    f.metadata = {'Cache-Control': 'no-cache'}
    content = await blocking_to_async(f.download_as_string())
    return content.decode('utf-8')


def exists_gs_file(bucket, path):
    bucket = gcs_client.bucket(bucket)
    f = bucket.blob(path)
    return f.exists()
