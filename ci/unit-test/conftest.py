"""Configure the unit-test environment before any ci.* modules are imported.

In CI these values come from the real deployment environment (env vars + /global-config volume
mount).  Locally they don't exist, so we set safe defaults and mock the config reader.
"""

import atexit
import json
import os
import unittest.mock

# Env vars read at module-import time by gear.profiling and ci.environment.
# setdefault leaves real CI values in place.
os.environ.setdefault('HAIL_SHA', 'test-sha')
os.environ.setdefault('HAIL_DEFAULT_NAMESPACE', 'default')
os.environ.setdefault('CLOUD', 'gcp')
os.environ.setdefault('HAIL_CI_UTILS_IMAGE', 'gcr.io/hail-vdc/ci-utils:test')
os.environ.setdefault('HAIL_BUILDKIT_IMAGE', 'gcr.io/hail-vdc/buildkit:test')
os.environ.setdefault('HAIL_CI_STORAGE_URI', 'gs://hail-ci-test/build')
os.environ.setdefault('HAIL_CI_GITHUB_CONTEXT', 'ci-test')

if not os.path.exists('/global-config'):
    # Patch gear.cloud_config.read_config_secret so that ci.environment's module-level
    # get_global_config() call (and the subsequent get_gcp_config() call) succeed locally.
    _fake_global_config = {
        'cloud': 'gcp',
        'docker_prefix': 'gcr.io/hail-vdc',
        'docker_root_image': 'ubuntu:22.04',
        'domain': 'hail.is',
        'kubernetes_server_url': 'https://k8s.example.com',
        'default_namespace': 'default',
        # Fields required by GCPConfig.from_global_config
        'batch_gcp_regions': json.dumps(['us-central1']),
        'gcp_region': 'us-central1',
        'gcp_project': 'hail-vdc',
        'gcp_zone': 'us-central1-a',
    }
    _patcher = unittest.mock.patch('gear.cloud_config.read_config_secret', return_value=_fake_global_config)
    _patcher.start()
    atexit.register(_patcher.stop)
