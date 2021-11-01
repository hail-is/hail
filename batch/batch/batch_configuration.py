import os
import json

KUBERNETES_TIMEOUT_IN_SECONDS = float(os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5.0))
REFRESH_INTERVAL_IN_SECONDS = int(os.environ.get('REFRESH_INTERVAL_IN_SECONDS', 5 * 60))
DEFAULT_NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']
PROJECT = os.environ['PROJECT']
SCOPE = os.environ['HAIL_SCOPE']

CLOUD = os.environ['CLOUD']
GCP_REGION = os.environ['HAIL_GCP_REGION']
GCP_ZONE = os.environ['HAIL_GCP_ZONE']
DOCKER_ROOT_IMAGE = os.environ['HAIL_DOCKER_ROOT_IMAGE']
DOCKER_PREFIX = os.environ['HAIL_DOCKER_PREFIX']

BATCH_GCP_REGIONS = set(json.loads(os.environ['HAIL_BATCH_GCP_REGIONS']))
BATCH_GCP_REGIONS.add(GCP_REGION)

assert PROJECT != ''
KUBERNETES_SERVER_URL = os.environ['KUBERNETES_SERVER_URL']
BATCH_STORAGE_URI = os.environ['HAIL_BATCH_STORAGE_URI']
HAIL_SHA = os.environ['HAIL_SHA']
HAIL_SHOULD_PROFILE = os.environ.get('HAIL_SHOULD_PROFILE') is not None
STANDING_WORKER_MAX_IDLE_TIME_MSECS = int(os.environ['STANDING_WORKER_MAX_IDLE_TIME_SECS']) * 1000
WORKER_MAX_IDLE_TIME_MSECS = 30 * 1000
HAIL_SHOULD_CHECK_INVARIANTS = os.environ.get('HAIL_SHOULD_CHECK_INVARIANTS') is not None

MACHINE_NAME_PREFIX = f'batch-worker-{DEFAULT_NAMESPACE}-'

# import os

# from gear.cloud_config import get_global_config

# global_config = get_global_config()

# CLOUD = global_config['cloud']
# DOCKER_ROOT_IMAGE = global_config['docker_root_image']
# DOCKER_PREFIX = global_config['docker_prefix']
# KUBERNETES_SERVER_URL = global_config['kubernetes_server_url']
# INTERNAL_GATEWAY_IP = global_config['internal_ip']

# <<<<<<< HEAD
# DEFAULT_NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']
# BATCH_BUCKET_NAME = os.environ['HAIL_BATCH_BUCKET_NAME']
# =======
# assert PROJECT != ''
# KUBERNETES_SERVER_URL = os.environ['KUBERNETES_SERVER_URL']
# BATCH_STORAGE_URI = os.environ['HAIL_BATCH_STORAGE_URI']
# >>>>>>> hi/main
# HAIL_SHA = os.environ['HAIL_SHA']
# SCOPE = os.environ['HAIL_SCOPE']

# KUBERNETES_TIMEOUT_IN_SECONDS = float(os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5.0))
# REFRESH_INTERVAL_IN_SECONDS = int(os.environ.get('REFRESH_INTERVAL_IN_SECONDS', 5 * 60))
# HAIL_SHOULD_PROFILE = os.environ.get('HAIL_SHOULD_PROFILE') is not None
# STANDING_WORKER_MAX_IDLE_TIME_MSECS = int(os.environ['STANDING_WORKER_MAX_IDLE_TIME_SECS']) * 1000
# WORKER_MAX_IDLE_TIME_MSECS = 30 * 1000
# HAIL_SHOULD_CHECK_INVARIANTS = os.environ.get('HAIL_SHOULD_CHECK_INVARIANTS') is not None

# MACHINE_NAME_PREFIX = f'batch-worker-{DEFAULT_NAMESPACE}-'
