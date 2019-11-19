import os

GCP_PROJECT = os.environ['HAIL_GCP_PROJECT']
GCP_ZONE = os.environ['HAIL_GCP_ZONE']
DOMAIN = os.environ['HAIL_DOMAIN']
IP = os.environ.get('HAIL_IP')
CI_UTILS_IMAGE = os.environ.get('HAIL_CI_UTILS_IMAGE', 'gcr.io/hail-vdc/ci-utils:latest')
SELF_HOSTNAME = os.environ.get('HAIL_SELF_HOSTNAME')
DEFAULT_NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']
BATCH_PODS_NAMESPACE = os.environ['HAIL_BATCH_PODS_NAMESPACE']
KUBERNETES_SERVER_URL = os.environ['KUBERNETES_SERVER_URL']
