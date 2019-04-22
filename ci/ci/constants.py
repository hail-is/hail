import os

GITHUB_API_URL = 'https://api.github.com/'
GITHUB_CLONE_URL = 'https://github.com/'
VERSION = '0-1'
BUILD_JOB_PREFIX = 'hail-ci-0-2-1'
BUILD_JOB_TYPE = BUILD_JOB_PREFIX + '-build'
DEPLOY_JOB_TYPE = BUILD_JOB_PREFIX + '-deploy'
GCP_PROJECT = 'broad-ctsa'
gcs_path = os.environ.get('HAIL_CI_GCS_PATH')
if gcs_path:
    path_pieces = gcs_path.split('/')
    GCS_BUCKET = path_pieces[0]
    if len(path_pieces) > 1:
        GCS_BUCKET_PREFIX = '/'.join(path_pieces[1:]) + '/'
    else:
        GCS_BUCKET_PREFIX = ''
else:
    GCS_BUCKET = 'hail-ci-' + VERSION
    GCS_BUCKET_PREFIX = ''
SHA_LENGTH = 12

BATCH_TEST_GSA_SECRET_NAME = os.environ.get('BATCH_TEST_GSA_SECRET_NAME', 'gsa-key-2x975')
BATCH_TEST_JWT_SECRET_NAME = os.environ.get('BATCH_TEST_JWT_SECRET_NAME', 'user-jwt-vkqfw')
