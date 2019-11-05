import os

KUBERNETES_TIMEOUT_IN_SECONDS = float(os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5.0))
REFRESH_INTERVAL_IN_SECONDS = int(os.environ.get('REFRESH_INTERVAL_IN_SECONDS', 5 * 60))
BATCH_NAMESPACE = os.environ.get('BATCH_NAMESPACE', 'default')
BATCH_WORKER_IMAGE = os.environ.get('BATCH_WORKER_IMAGE', 'gcr.io/hail-vdc/batch-worker:latest')
PROJECT = os.environ.get('PROJECT', 'hail-vdc')
ZONE = os.environ.get('ZONE', 'us-central1-a')
WORKER_TYPE = os.environ.get('WORKER_TYPE', 'standard')
WORKER_CORES = int(os.environ.get('WORKER_CORES', 8))
WORKER_DISK_SIZE_GB = int(os.environ.get('WORKER_DISK_SIZE_GB', 100))
POOL_SIZE = int(os.environ.get('POOL_SIZE', 100))
MAX_INSTANCES = int(os.environ.get('MAX_INSTANCES', 5))
