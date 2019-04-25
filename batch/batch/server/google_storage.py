

def upload_private_gs_file_from_string(gcs_client, bucket, target_path, string):
    bucket = gcs_client.bucket(bucket)
    f = bucket.blob(target_path)
    f.metadata = {'Cache-Control': 'no-cache'}
    f.upload_from_string(string)


def download_gs_file_as_string(gcs_client, bucket, path):
    bucket = gcs_client.bucket(bucket)
    f = bucket.blob(path)
    f.metadata = {'Cache-Control': 'no-cache'}
    content = f.download_as_string()
    return content.decode('utf-8')


def delete_gs_file(gcs_client, bucket, path):
    bucket = gcs_client.bucket(bucket)
    f = bucket.blob(path)
    f.delete()
