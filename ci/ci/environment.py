import os

GCP_PROJECT = os.environ['HAIL_GCP_PROJECT']
assert GCP_PROJECT != ''
GCP_ZONE = os.environ['HAIL_GCP_ZONE']
assert GCP_ZONE != ''
GCP_REGION = '-'.join(GCP_ZONE.split('-')[:-1])  # us-west1-a -> us-west1
DOCKER_PREFIX = os.environ.get('HAIL_DOCKER_PREFIX', f'gcr.io/{GCP_REGION}')
assert DOCKER_PREFIX != ''
DOCKER_ROOT_IMAGE = os.environ['HAIL_DOCKER_ROOT_IMAGE']
assert DOCKER_ROOT_IMAGE != ''
DOMAIN = os.environ['HAIL_DOMAIN']
assert DOMAIN != ''
IP = os.environ.get('HAIL_IP')
CI_UTILS_IMAGE = os.environ.get('HAIL_CI_UTILS_IMAGE', f'{DOCKER_PREFIX}/ci-utils:latest')
BUILDKIT_IMAGE = os.environ['HAIL_BUILDKIT_IMAGE']
DEFAULT_NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']
KUBERNETES_SERVER_URL = os.environ['KUBERNETES_SERVER_URL']
BUCKET = os.environ['HAIL_CI_BUCKET_NAME']
