import json
import os

from gear.cloud_config import get_azure_config, get_gcp_config, get_global_config

global_config = get_global_config()

CLOUD = global_config['cloud']
assert CLOUD in ('gcp', 'azure'), CLOUD

DOCKER_PREFIX = global_config['docker_prefix']
DOCKER_ROOT_IMAGE = global_config['docker_root_image']
DOMAIN = global_config['domain']
KUBERNETES_SERVER_URL = global_config['kubernetes_server_url']
DEFAULT_NAMESPACE = global_config['default_namespace']

CI_UTILS_IMAGE = os.environ['HAIL_CI_UTILS_IMAGE']
BUILDKIT_IMAGE = os.environ['HAIL_BUILDKIT_IMAGE']
STORAGE_URI = os.environ['HAIL_CI_STORAGE_URI']
DEPLOY_STEPS = tuple(json.loads(os.environ.get('HAIL_CI_DEPLOY_STEPS', '[]')))

if CLOUD == 'gcp':
    REGION = get_gcp_config().region
else:
    assert CLOUD == 'azure', CLOUD
    REGION = get_azure_config().region
