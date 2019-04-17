import os

import kubernetes as kube
import googleapiclient.discovery

if 'BATCH_USE_KUBE_CONFIG' in os.environ:
    kube.config.load_kube_config()
else:
    kube.config.load_incluster_config()

kube_client = kube.client
v1 = kube.client.CoreV1Api()

if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/gsa-key/privateKeyData'

gcloud_service = googleapiclient.discovery.build('iam', 'v1')
