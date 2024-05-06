import pytest

from batch.cloud.gcp.resource_utils import gcp_worker_memory_per_core_mib, machine_type_to_gpu_num
from batch.cloud.gcp.resources import GCPAcceleratorResource, gcp_resource_from_dict
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


def test_gcp_resource_from_dict():
    name = 'accelerator/l4-nonpreemptible/us-central1/1712657549063'
    gpu_data_dic_single = {'name': name, 'number': 1, 'type': 'gcp_accelerator', 'format_version': 2}
    resource = gcp_resource_from_dict(gpu_data_dic_single)
    quantified_resources = resource.to_quantified_resource(1000, 20, 1024, 20)
    assert quantified_resources
    assert quantified_resources['quantity'] == 1024

    gpu_data_dic_double = {'name': name, 'number': 2, 'type': 'gcp_accelerator', 'format_version': 2}
    resource = gcp_resource_from_dict(gpu_data_dic_double)
    quantified_resources = resource.to_quantified_resource(1000, 20, 1024, 20)
    assert quantified_resources
    assert quantified_resources['quantity'] == 2048


def test_machine_type_to_gpu_num():
    assert machine_type_to_gpu_num('g2-standard-4') == 1
    assert machine_type_to_gpu_num('g2-standard-8') == 1
    assert machine_type_to_gpu_num('g2-standard-12') == 1
    assert machine_type_to_gpu_num('g2-standard-16') == 1
    assert machine_type_to_gpu_num('g2-standard-32') == 1
    assert machine_type_to_gpu_num('g2-standard-24') == 2
    assert machine_type_to_gpu_num('g2-standard-48') == 4
    assert machine_type_to_gpu_num('g2-standard-96') == 8


def test_gcp_accelerator_to_from_dict():
    version_1_dict = {
        'type': 'gcp_accelerator',
        'name': 'accelerator/l4-nonpreemptible/us-central1/1712657549063',
        'format_version': 1,
    }
    version_1_resource = GCPAcceleratorResource.from_dict(version_1_dict)
    assert version_1_resource
    version_1_remade_dict = version_1_resource.to_dict()
    del version_1_remade_dict['number']
    assert version_1_remade_dict == version_1_dict

    version_2_dict = {
        'type': 'gcp_accelerator',
        'name': 'accelerator/l4-nonpreemptible/us-central1/1712657549063',
        'format_version': 2,
        'number': 2,
    }
    version_2_resource = GCPAcceleratorResource.from_dict(version_2_dict)
    assert version_2_resource
    version_2_remade_dict = version_2_resource.to_dict()
    assert version_2_remade_dict == version_2_dict
