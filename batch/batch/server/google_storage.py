from google.cloud import storage


gcs_client = storage.Client()


def upload_private_gs_file_from_string(bucket, target_path, string):
    bucket = gcs_client.bucket(bucket)
    f = bucket.blob(target_path)
    f.metadata = {'Cache-Control': 'no-cache'}
    f.upload_from_string(string)


def download_gs_file_as_string(bucket, path):
    bucket = gcs_client.bucket(bucket)
    f = bucket.blob(path)
    f.metadata = {'Cache-Control': 'no-cache'}
    content = f.download_as_string()
    return content.decode('utf-8')


def exists_gs_file(bucket, path):
    bucket = gcs_client.bucket(bucket)
    f = bucket.blob(path)
    return f.exists()
