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


@pytest.mark.parametrize(
    "image,expected",
    [
        # Bare images (should be rewritten)
        ("ubuntu:20.04", "us-central1-docker.pkg.dev/my-project/dockerhubproxy/library/ubuntu:20.04"),
        ("ubuntu", "us-central1-docker.pkg.dev/my-project/dockerhubproxy/library/ubuntu"),
        ("python:3.9", "us-central1-docker.pkg.dev/my-project/dockerhubproxy/library/python:3.9"),
        # Namespaced images (should be rewritten)
        ("myorg/myimage:tag", "us-central1-docker.pkg.dev/my-project/dockerhubproxy/myorg/myimage:tag"),
        ("envoyproxy/envoy:v1.33.0", "us-central1-docker.pkg.dev/my-project/dockerhubproxy/envoyproxy/envoy:v1.33.0"),
        # Images with explicit registry (should NOT be rewritten)
        ("gcr.io/myproject/image:tag", None),
        ("us-central1-docker.pkg.dev/project/repo/image", None),
        ("myregistry.io/image:tag", None),
        ("localhost:5000/image", None),
        ("registry.example.com:8080/image", None),
        # Edge cases
        (
            "image.with.dots:tag",
            "us-central1-docker.pkg.dev/my-project/dockerhubproxy/library/image.with.dots:tag",
        ),  # dots in first part
        (
            "image:with:colons",
            "us-central1-docker.pkg.dev/my-project/dockerhubproxy/library/image:with:colons",
        ),  # colons in first part
        (
            "my-org/my-image:1.0.0",
            "us-central1-docker.pkg.dev/my-project/dockerhubproxy/my-org/my-image:1.0.0",
        ),  # hyphens OK
    ],
)
def test_rewrite_dockerhub_image(image, expected):
    dockerhub_prefix = "us-central1-docker.pkg.dev/my-project/dockerhubproxy"
    assert rewrite_dockerhub_image(image, dockerhub_prefix) == expected
