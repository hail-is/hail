import pytest

from batch.cloud.gcp.resource_utils import gcp_worker_memory_per_core_mib
from batch.cloud.resource_utils import adjust_cores_for_packability
from hailtop.batch_client.parse import parse_memory_in_bytes


def test_packability():
    assert adjust_cores_for_packability(0) == 250
    assert adjust_cores_for_packability(200) == 250
    assert adjust_cores_for_packability(250) == 250
    assert adjust_cores_for_packability(251) == 500
    assert adjust_cores_for_packability(500) == 500
    assert adjust_cores_for_packability(501) == 1000
    assert adjust_cores_for_packability(1000) == 1000
    assert adjust_cores_for_packability(1001) == 2000
    assert adjust_cores_for_packability(2000) == 2000
    assert adjust_cores_for_packability(2001) == 4000
    assert adjust_cores_for_packability(3000) == 4000
    assert adjust_cores_for_packability(4000) == 4000
    assert adjust_cores_for_packability(4001) == 8000
    assert adjust_cores_for_packability(8001) == 16000


def test_memory_str_to_bytes():
    assert parse_memory_in_bytes('7') == 7
    assert parse_memory_in_bytes('1K') == 1000
    assert parse_memory_in_bytes('1Ki') == 1024


def test_gcp_worker_memory_per_core_mib():
    with pytest.raises(AssertionError):
        assert gcp_worker_memory_per_core_mib('n2', 'standard')
    assert gcp_worker_memory_per_core_mib('n1', 'standard') == 3840
    assert gcp_worker_memory_per_core_mib('n1', 'highmem') == 6656
    assert gcp_worker_memory_per_core_mib('n1', 'highcpu') == 924
    assert gcp_worker_memory_per_core_mib('g2', 'standard') == 4000
