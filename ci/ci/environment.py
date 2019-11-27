import os

GCP_PROJECT = os.environ['HAIL_GCP_PROJECT']
assert GCP_PROJECT != ''
GCP_ZONE = os.environ['HAIL_GCP_ZONE']
assert GCP_ZONE != ''
DOMAIN = os.environ['HAIL_DOMAIN']
assert DOMAIN != ''
IP = os.environ.get('HAIL_IP')
CI_UTILS_IMAGE = os.environ.get('HAIL_CI_UTILS_IMAGE', 'gcr.io/hail-vdc/ci-utils:latest')
DEFAULT_NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']
BATCH_PODS_NAMESPACE = os.environ['HAIL_BATCH_PODS_NAMESPACE']
KUBERNETES_SERVER_URL = os.environ['KUBERNETES_SERVER_URL']
