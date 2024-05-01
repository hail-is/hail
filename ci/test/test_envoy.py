import os
from pathlib import Path

import pytest
import yaml

from ci.envoy import create_cds_response, create_rds_response


@pytest.mark.parametrize(
    'proxy, create_xds_response, expected_config',
    [
        ('gateway', create_cds_response, 'test_gateway_cds_response.yaml'),
        ('gateway', create_rds_response, 'test_gateway_rds_response.yaml'),
        ('internal-gateway', create_cds_response, 'test_internal_gateway_cds_response.yaml'),
        ('internal-gateway', create_rds_response, 'test_internal_gateway_rds_response.yaml'),
    ],
)
def test_gateway_cds_generation(proxy, create_xds_response, expected_config):
    with open(Path(os.path.dirname(__file__)) / 'envoy' / expected_config, encoding='utf-8') as f:
        expected_cluster_config = yaml.safe_load(f.read())
    actual_cluster_config = create_xds_response(['foo'], {'test': ['bar']}, proxy)
    assert expected_cluster_config == actual_cluster_config
