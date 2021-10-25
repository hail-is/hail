import os

from gear.cloud_config import get_global_config

global_config = get_global_config()

CLOUD = global_config['cloud']
DOCKER_ROOT_IMAGE = global_config['docker_root_image']
DOCKER_PREFIX = global_config['docker_prefix']
KUBERNETES_SERVER_URL = global_config['kubernetes_server_url']
INTERNAL_GATEWAY_IP = global_config['internal_ip']

DEFAULT_NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']
BATCH_BUCKET_NAME = os.environ['HAIL_BATCH_BUCKET_NAME']
HAIL_SHA = os.environ['HAIL_SHA']
SCOPE = os.environ['HAIL_SCOPE']

KUBERNETES_TIMEOUT_IN_SECONDS = float(os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5.0))
REFRESH_INTERVAL_IN_SECONDS = int(os.environ.get('REFRESH_INTERVAL_IN_SECONDS', 5 * 60))
HAIL_SHOULD_PROFILE = os.environ.get('HAIL_SHOULD_PROFILE') is not None
STANDING_WORKER_MAX_IDLE_TIME_MSECS = int(os.environ['STANDING_WORKER_MAX_IDLE_TIME_SECS']) * 1000
WORKER_MAX_IDLE_TIME_MSECS = 30 * 1000
HAIL_SHOULD_CHECK_INVARIANTS = os.environ.get('HAIL_SHOULD_CHECK_INVARIANTS') is not None

MACHINE_NAME_PREFIX = f'batch-worker-{DEFAULT_NAMESPACE}-'
