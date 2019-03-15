import os

import kubernetes as kube

from google.oauth2 import service_account
from google.cloud import storage
import googleapiclient.discovery

if 'BATCH_USE_KUBE_CONFIG' in os.environ:
    kube.config.load_kube_config()
else:
    kube.config.load_incluster_config()

kube_client = kube.client
k8s = kube.client.CoreV1Api()

# Get credentials
credentials = service_account.Credentials.from_service_account_file(
    filename=os.environ['GOOGLE_APPLICATION_CREDENTIALS'],
    scopes=['https://www.googleapis.com/auth/cloud-platform'])

# Create the Cloud IAM service object
gcloud_service = googleapiclient.discovery.build(
    'iam', 'v1', credentials=credentials)