import pytest

from batch.cloud.azure.resource_utils import MACHINE_TYPE_TO_PARTS as MACHINE_TYPE_TO_PARTS_AZURE
from batch.cloud.gcp.resource_utils import MACHINE_TYPE_TO_PARTS as MACHINE_TYPE_TO_PARTS_GCP
from batch.cloud.gcp.resource_utils import gcp_worker_memory_per_core_mib, machine_type_to_gpu_num
from batch.cloud.gcp.resources import GCPAcceleratorResource, gcp_resource_from_dict
from batch.cloud.resource_utils import adjust_cores_for_packability
from batch.utils import rewrite_dockerhub_image
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


def test_gcp_machine_memory_per_core_mib():
    for _, machine_parts in MACHINE_TYPE_TO_PARTS_GCP.items():
        if machine_parts.machine_family == 'n1' and machine_parts.worker_type == 'standard':
            assert int(machine_parts.memory / machine_parts.cores / 1024**2) == 3840
        elif machine_parts.machine_family == 'n1' and machine_parts.worker_type == 'highmem':
            assert int(machine_parts.memory / machine_parts.cores / 1024**2) == 6656
        elif machine_parts.machine_family == 'n1' and machine_parts.worker_type == 'highcpu':
            assert int(machine_parts.memory / machine_parts.cores / 1024**2) == 924
        elif machine_parts.machine_family == 'g2' and machine_parts.worker_type == 'standard':
            assert int(machine_parts.memory / machine_parts.cores / 1024**2) == 4096
        elif machine_parts.machine_family == 'a2' and machine_parts.worker_type == 'highgpu':
            assert machine_parts.gpu_config
            assert int(machine_parts.memory / machine_parts.gpu_config.num_gpus / 1024**3) == 85
        elif machine_parts.machine_family == 'a2' and machine_parts.worker_type == 'megagpu':
            assert machine_parts.gpu_config
            assert int(machine_parts.memory / machine_parts.gpu_config.num_gpus / 1024**3) == 85
        elif machine_parts.machine_family == 'a2' and machine_parts.worker_type == 'ultragpu':
            assert machine_parts.gpu_config
            assert int(machine_parts.memory / machine_parts.gpu_config.num_gpus / 1024**3) == 170


def test_azure_machine_memory_per_core_mib():
    for _, machine_parts in MACHINE_TYPE_TO_PARTS_AZURE.items():
        if machine_parts.family == 'F':
            assert int(machine_parts.memory / machine_parts.cores / 1024**2) == 2048
        elif machine_parts.family == 'D':
            assert int(machine_parts.memory / machine_parts.cores / 1024**2) == 4096
        elif machine_parts.family == 'E':
            assert int(machine_parts.memory / machine_parts.cores / 1024**2) == 8192


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
    assert version_1_remade_dict['number'] == 1

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


def test_rewrite_dockerhub_image():
    dockerhub_prefix = "us-central1-docker.pkg.dev/my-project/dockerhubproxy"

    # Test bare images (should be rewritten)
    assert rewrite_dockerhub_image("ubuntu:20.04", dockerhub_prefix) == f"{dockerhub_prefix}/library/ubuntu:20.04"
    assert rewrite_dockerhub_image("ubuntu", dockerhub_prefix) == f"{dockerhub_prefix}/library/ubuntu"
    assert rewrite_dockerhub_image("python:3.9", dockerhub_prefix) == f"{dockerhub_prefix}/library/python:3.9"

    # Test namespaced images (should be rewritten)
    assert rewrite_dockerhub_image("myorg/myimage:tag", dockerhub_prefix) == f"{dockerhub_prefix}/myorg/myimage:tag"
    assert (
        rewrite_dockerhub_image("envoyproxy/envoy:v1.33.0", dockerhub_prefix)
        == f"{dockerhub_prefix}/envoyproxy/envoy:v1.33.0"
    )

    # Test images with explicit registry (should NOT be rewritten)
    assert rewrite_dockerhub_image("gcr.io/myproject/image:tag", dockerhub_prefix) is None
    assert rewrite_dockerhub_image("us-central1-docker.pkg.dev/project/repo/image", dockerhub_prefix) is None
    assert rewrite_dockerhub_image("myregistry.io/image:tag", dockerhub_prefix) is None
    assert rewrite_dockerhub_image("localhost:5000/image", dockerhub_prefix) is None
    assert rewrite_dockerhub_image("registry.example.com:8080/image", dockerhub_prefix) is None

    # Test edge cases
    assert (
        rewrite_dockerhub_image("image.with.dots:tag", dockerhub_prefix)
        == f"{dockerhub_prefix}/library/image.with.dots:tag"
    )  # dots in first part
    assert (
        rewrite_dockerhub_image("image:with:colons", dockerhub_prefix)
        == f"{dockerhub_prefix}/library/image:with:colons"
    )  # colons in first part
    assert (
        rewrite_dockerhub_image("my-org/my-image:1.0.0", dockerhub_prefix)
        == f"{dockerhub_prefix}/my-org/my-image:1.0.0"
    )  # hyphens OK
