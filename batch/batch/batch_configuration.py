import os
import uuid

KUBERNETES_TIMEOUT_IN_SECONDS = float(os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5.0))
REFRESH_INTERVAL_IN_SECONDS = int(os.environ.get('REFRESH_INTERVAL_IN_SECONDS', 2 * 60))
HAIL_POD_NAMESPACE = os.environ.get('HAIL_POD_NAMESPACE', 'batch-pods')
POD_VOLUME_SIZE = os.environ.get('POD_VOLUME_SIZE', '10Mi')
INSTANCE_ID = os.environ.get('HAIL_INSTANCE_ID', uuid.uuid4().hex)
BATCH_IMAGE = os.environ.get('BATCH_IMAGE', 'gcr.io/hail-vdc/batch:latest')
QUEUE_SIZE = os.environ.get('QUEUE_SIZE', 1_000_000)
MAX_PODS = os.environ.get('MAX_PODS', 30_000)
