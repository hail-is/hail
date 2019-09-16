import os

GCP_PROJECT = os.environ.get('HAIL_GCP_PROJECT', 'hail-vdc')
DOMAIN = os.environ['HAIL_DOMAIN']
IP = os.environ.get('HAIL_IP')
CI_UTILS_IMAGE = os.environ.get('HAIL_CI_UTILS_IMAGE', 'gcr.io/hail-vdc/ci-utils:latest')
SELF_HOSTNAME = os.environ.get('HAIL_SELF_HOSTNAME')
