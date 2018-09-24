from constants import GCP_PROJECT, VERSION
from google.cloud import storage
import os

# this is a bit of a hack, but makes my development life easier
if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
        'gcloud-token/hail-ci-' + VERSION + '.key'
gcs_client = storage.Client(project=GCP_PROJECT)


def upload_public_gs_file_from_string(bucket, target_path, string):
    create_public_gs_file(bucket,
                          target_path,
                          lambda f: f.upload_from_string(string))


def upload_public_gs_file_from_filename(bucket, target_path, filename):
    create_public_gs_file(bucket,
                          target_path,
                          lambda f: f.upload_from_filename(filename))


def create_public_gs_file(bucket, target_path, upload):
    bucket = gcs_client.bucket(bucket)
    f = bucket.blob(target_path)
    # https://developers.google.com/web/fundamentals/performance/optimizing-content-efficiency/http-caching
    f.metadata = {'Cache-Control': 'no-cache'}
    upload(f)
    f.acl.all().grant_read()
    f.acl.save()
