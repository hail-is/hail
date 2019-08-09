import os
import uuid

KUBERNETES_TIMEOUT_IN_SECONDS = float(os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5.0))
REFRESH_INTERVAL_IN_SECONDS = int(os.environ.get('REFRESH_INTERVAL_IN_SECONDS', 5 * 60))
BATCH_NAMESPACE = os.environ.get('BATCH_NAMESPACE', 'default')
POD_VOLUME_SIZE = os.environ.get('POD_VOLUME_SIZE', '10Mi')
INSTANCE_ID = os.environ.get('HAIL_INSTANCE_ID', uuid.uuid4().hex)
BATCH_IMAGE = os.environ.get('BATCH_IMAGE', 'gcr.io/hail-vdc/batch:latest')
PROJECT = os.environ.get('PROJECT', 'hail-vdc')
ZONE = os.environ.get('ZONE', 'us-central1-a')
