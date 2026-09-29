from typing import Optional

import pytest

from batch.cloud.azure.resource_utils import MACHINE_TYPE_TO_PARTS as MACHINE_TYPE_TO_PARTS_AZURE
from batch.cloud.gcp.instance_config import GCPSlimInstanceConfig, region_from_location
from batch.cloud.gcp.resource_utils import (
    GCP_HYPERDISK_BALANCED_FREE_IOPS,
    GCP_HYPERDISK_BALANCED_FREE_THROUGHPUT_MIB_PER_SEC,
    gcp_boot_disk_type,
    gcp_data_disk_device_name,
    gcp_data_disk_type,
    gcp_hyperdisk_performance_overrides,
    gcp_local_ssd_count,
    gcp_local_ssd_size,
    gcp_worker_memory_per_core_mib,
    machine_type_to_gpu_num,
)
from batch.cloud.gcp.resource_utils import (
    MACHINE_TYPE_TO_PARTS as MACHINE_TYPE_TO_PARTS_GCP,
)
from batch.cloud.gcp.resources import GCPAcceleratorResource, gcp_resource_from_dict
from batch.cloud.resource_utils import adjust_cores_for_packability
from batch.driver.billing_manager import ProductVersions
from batch.driver.exceptions import LocalSSDNotSupportedError
from batch.driver.naming import build_inst_coll_regex, make_machine_name
from batch.preemption_costs import (
    AttemptResource,
    nonpreemptible_counterparts,
    nonpreemptible_product,
    projected_nonpreemptible_cost,
    rate_in_effect,
    retried_attempts_cost,
)
from batch.utils import rewrite_dockerhub_image
from hailtop.batch_client.parse import parse_memory_in_bytes
from hailtop.utils import secret_alnum_string


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
    assert gcp_worker_memory_per_core_mib('n1', 'standard') == 3840
    assert gcp_worker_memory_per_core_mib('n1', 'highmem') == 6656
    assert gcp_worker_memory_per_core_mib('n1', 'highcpu') == 924
    assert gcp_worker_memory_per_core_mib('n2', 'standard') == 4096
    assert gcp_worker_memory_per_core_mib('n2', 'highmem') == 8192
    assert gcp_worker_memory_per_core_mib('n2', 'highcpu') == 1024
    assert gcp_worker_memory_per_core_mib('n4', 'standard') == 4096
    assert gcp_worker_memory_per_core_mib('n4', 'highmem') == 8192
    assert gcp_worker_memory_per_core_mib('n4', 'highcpu') == 2048


def test_gcp_machine_memory_per_core_mib():
    for _, machine_parts in MACHINE_TYPE_TO_PARTS_GCP.items():
        if machine_parts.machine_family == 'n1' and machine_parts.worker_type == 'standard':
            assert int(machine_parts.memory / machine_parts.cores / 1024**2) == 3840
        elif machine_parts.machine_family == 'n1' and machine_parts.worker_type == 'highmem':
            assert int(machine_parts.memory / machine_parts.cores / 1024**2) == 6656
        elif machine_parts.machine_family == 'n1' and machine_parts.worker_type == 'highcpu':
            assert int(machine_parts.memory / machine_parts.cores / 1024**2) == 924
        elif machine_parts.machine_family == 'n2' and machine_parts.worker_type == 'standard':
            assert int(machine_parts.memory / machine_parts.cores / 1024**2) == 4096
        elif machine_parts.machine_family == 'n2' and machine_parts.worker_type == 'highmem':
            if machine_parts.cores == 128:
                assert int(machine_parts.memory / machine_parts.cores / 1024**2) == 6912
            else:
                assert int(machine_parts.memory / machine_parts.cores / 1024**2) == 8192
        elif machine_parts.machine_family == 'n2' and machine_parts.worker_type == 'highcpu':
            assert int(machine_parts.memory / machine_parts.cores / 1024**2) == 1024
        elif machine_parts.machine_family == 'n4' and machine_parts.worker_type == 'standard':
            assert int(machine_parts.memory / machine_parts.cores / 1024**2) == 4096
        elif machine_parts.machine_family == 'n4' and machine_parts.worker_type == 'highmem':
            assert int(machine_parts.memory / machine_parts.cores / 1024**2) == 8192
        elif machine_parts.machine_family == 'n4' and machine_parts.worker_type == 'highcpu':
            assert int(machine_parts.memory / machine_parts.cores / 1024**2) == 2048
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


@pytest.mark.parametrize(
    "family,cores,expected",
    [
        ('n1', 16, 1),
        ('n1', 96, 1),
        ('n2', 2, 1),
        ('n2', 4, 1),
        ('n2', 8, 1),
        ('n2', 16, 2),
        ('n2', 32, 4),
        ('n2', 48, 8),
        ('n2', 64, 8),
        ('n2', 80, 8),
        ('n2', 96, 16),
        ('n2', 128, 16),
    ],
)
def test_gcp_local_ssd_count(family, cores, expected):
    assert gcp_local_ssd_count(family, cores) == expected


def test_gcp_local_ssd_count_rejects_n4():
    # n4 supports zero local SSDs; it must never fall through to the generic non-n2 default of 1.
    with pytest.raises(LocalSSDNotSupportedError):
        gcp_local_ssd_count('n4', 16)


def test_gcp_instance_config_rejects_local_ssd_on_n4():
    # An unprovisionable combination must be rejected where the config is built, before any
    # billing resources exist for it. The empty ProductVersions asserts that: no product
    # lookup happens before the check.
    with pytest.raises(LocalSSDNotSupportedError):
        GCPSlimInstanceConfig.create(
            product_versions=ProductVersions({}),
            machine_type='n4-standard-16',
            preemptible=False,
            local_ssd_data_disk=True,
            data_disk_size_gb=375,
            boot_disk_size_gb=30,
            job_private=False,
            location='us-central1-a',
        )


def test_gcp_instance_config_rejects_unknown_machine_type():
    with pytest.raises(ValueError, match='bad machine_type'):
        GCPSlimInstanceConfig.create(
            product_versions=ProductVersions({}),
            machine_type='n4-nonsense-16',
            preemptible=False,
            local_ssd_data_disk=False,
            data_disk_size_gb=375,
            boot_disk_size_gb=30,
            job_private=False,
            location='us-central1-a',
        )


def test_gcp_disk_type_helpers():
    assert gcp_boot_disk_type('n4') == 'hyperdisk-balanced'
    assert gcp_boot_disk_type('n2') == 'pd-ssd'
    assert gcp_boot_disk_type('n1') == 'pd-ssd'

    assert gcp_data_disk_type('n4') == 'hyperdisk-balanced'
    assert gcp_data_disk_type('n2') == 'pd-ssd'
    assert gcp_data_disk_type('n1') == 'pd-ssd'

    assert gcp_data_disk_device_name('n4', 'n4-standard-16') == 'nvme0n2'
    assert gcp_data_disk_device_name('g2', 'g2-standard-4') == 'nvme0n2'
    assert gcp_data_disk_device_name('n2', 'n2-standard-16') == 'sdb'
    assert gcp_data_disk_device_name('n1', 'n1-standard-16') == 'sdb'


def test_gcp_hyperdisk_performance_overrides_pin_free_baseline():
    overrides = gcp_hyperdisk_performance_overrides('hyperdisk-balanced')
    assert overrides == {
        'provisionedIops': str(GCP_HYPERDISK_BALANCED_FREE_IOPS),
        'provisionedThroughput': str(GCP_HYPERDISK_BALANCED_FREE_THROUGHPUT_MIB_PER_SEC),
    }
    assert GCP_HYPERDISK_BALANCED_FREE_IOPS == 3000
    assert GCP_HYPERDISK_BALANCED_FREE_THROUGHPUT_MIB_PER_SEC == 140


def test_gcp_hyperdisk_performance_overrides_noop_for_non_hyperdisk():
    # provisionedIops/provisionedThroughput are rejected by the GCE API for non-Hyperdisk disk
    # types, so no fields should be added for e.g. pd-ssd.
    assert not gcp_hyperdisk_performance_overrides('pd-ssd')


@pytest.mark.parametrize(
    "family,cores,expected",
    [
        ('n1', 16, 375),
        ('n2', 2, 375),
        ('n2', 16, 750),
        ('n2', 48, 3000),
        ('n2', 128, 6000),
    ],
)
def test_gcp_local_ssd_size(family, cores, expected):
    assert gcp_local_ssd_size(family, cores) == expected


@pytest.mark.parametrize(
    "location,expected",
    [
        ('us-central1', 'us-central1'),
        ('us-east1', 'us-east1'),
        ('northamerica-northeast1', 'northamerica-northeast1'),
        ('us-central1-a', 'us-central1'),
        ('us-central1-b', 'us-central1'),
        ('northamerica-northeast1-a', 'northamerica-northeast1'),
    ],
)
def test_region_from_location(location, expected):
    assert region_from_location(location) == expected


@pytest.mark.parametrize(
    "location",
    [
        '',
        'uscentral1',
        'us-central1-a-b',
        'us-central1-a-b-c',
    ],
)
def test_region_from_location_rejects_malformed(location):
    with pytest.raises(ValueError, match='Expected a GCP region or zone'):
        region_from_location(location)


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


_INST_COLL_NAMES = [
    'standard',
    'highmem',
    'lowmem',
    'standard-np',  # hyphenated
    'pool-abcde',  # last segment looks like an old 5-char suffix
    'pool-abcdef',  # last segment looks like half the old 6-6 suffix
    'pool-abcdef-ghijkl',  # last two segments look like the old 6-6 suffix
    'pool-abcde-fghij',  # two 5-char segments
]


@pytest.mark.parametrize('inst_coll_name', _INST_COLL_NAMES)
def test_machine_name_inst_coll_roundtrip(inst_coll_name):
    manager_prefix = 'batch-worker-default-'
    child_prefix = f'{manager_prefix}{inst_coll_name}-'
    machine_name = make_machine_name(child_prefix)
    assert len(machine_name) <= 63, f'machine name exceeds GCE limit: {machine_name!r}'
    match = build_inst_coll_regex(manager_prefix).search(machine_name)
    assert match is not None, f'regex did not match {machine_name!r}'
    assert match.group('inst_coll') == inst_coll_name


@pytest.mark.parametrize('inst_coll_name', _INST_COLL_NAMES)
def test_machine_name_inst_coll_roundtrip_long_namespace(inst_coll_name):
    # Verify names stay within GCE's 63-char limit even with long namespaces (e.g. CI test namespaces).
    ns = secret_alnum_string(20, case='lower')
    manager_prefix = f'batch-worker-{ns}-'
    child_prefix = f'{manager_prefix}{inst_coll_name}-'
    machine_name = make_machine_name(child_prefix)
    assert len(machine_name) <= 63, f'machine name exceeds GCE limit: {machine_name!r}'
    match = build_inst_coll_regex(manager_prefix).search(machine_name)
    assert match is not None, f'regex did not match {machine_name!r}'
    assert match.group('inst_coll') == inst_coll_name


@pytest.mark.parametrize('inst_coll_name', _INST_COLL_NAMES)
def test_old_style_machine_name_inst_coll_roundtrip(inst_coll_name):
    manager_prefix = 'batch-worker-default-'
    machine_name = f'{manager_prefix}{inst_coll_name}-ab1cd'  # fixed 5-char alphanumeric suffix
    match = build_inst_coll_regex(manager_prefix).search(machine_name)
    assert match is not None, f'regex did not match {machine_name!r}'
    assert match.group('inst_coll') == inst_coll_name


@pytest.mark.parametrize(
    'product,expected',
    [
        ('compute/n1-preemptible/us-central1', 'compute/n1-nonpreemptible/us-central1'),
        ('memory/n4-preemptible/us-central1', 'memory/n4-nonpreemptible/us-central1'),
        ('accelerator/l4-preemptible/us-central1', 'accelerator/l4-nonpreemptible/us-central1'),
        ('compute/n1-preemptible', 'compute/n1-nonpreemptible'),
        ('disk/local-ssd/preemptible/us-central1', 'disk/local-ssd/nonpreemptible/us-central1'),
        ('ip-fee/preemptible/1024', 'ip-fee/nonpreemptible/1024'),
    ],
)
def test_nonpreemptible_product_of_preemptible_product(product, expected):
    assert nonpreemptible_product(product) == expected


@pytest.mark.parametrize(
    'product',
    [
        'compute/n1-nonpreemptible/us-central1',
        'disk/local-ssd/nonpreemptible/us-central1',
        'disk/pd-ssd/us-central1',
        'disk/hyperdisk-balanced/us-central1',
        'service-fee',
        'disk/local-ssd',
        'ip-fee/1024',
    ],
)
def test_nonpreemptible_product_passes_through_other_products(product):
    assert nonpreemptible_product(product) is None


def test_rate_in_effect_compares_versions_numerically():
    resources = [
        ('compute/n1-nonpreemptible/us-central1/999', 1.0),
        ('compute/n1-nonpreemptible/us-central1/1000', 2.0),
        ('compute/n1-nonpreemptible/us-central1/2000', 3.0),
    ]
    assert rate_in_effect(resources, 'compute/n1-nonpreemptible/us-central1', 1500) == 2.0
    assert rate_in_effect(resources, 'compute/n1-nonpreemptible/us-central1', 1000) == 2.0
    assert rate_in_effect(resources, 'compute/n1-nonpreemptible/us-central1', 999) == 1.0


def test_rate_in_effect_legacy_version_is_always_in_effect():
    assert rate_in_effect([('compute/n1-nonpreemptible/1', 5.0)], 'compute/n1-nonpreemptible', 1) == 5.0


def test_rate_in_effect_is_none_when_no_version_is_in_effect_yet():
    resources = [('compute/n1-nonpreemptible/us-central1/2000', 3.0)]
    assert rate_in_effect(resources, 'compute/n1-nonpreemptible/us-central1', 1999) is None
    assert rate_in_effect([], 'compute/n1-nonpreemptible/us-central1', 1999) is None


def test_rate_in_effect_only_matches_the_exact_product():
    resources = [('compute/n1-nonpreemptible/us-central1/1000', 3.0)]
    assert rate_in_effect(resources, 'compute/n1-nonpreemptible', 2000) is None


def _attempt_resource(
    attempt_id: str,
    reason: str,
    start_time: Optional[int] = 0,
    rollup_time: Optional[int] = 1000,
    resource: str = 'compute/n1-preemptible/us-central1/1',
    rate: float = 0.5,
    quantity: int = 2,
) -> AttemptResource:
    return AttemptResource(
        attempt_id=attempt_id,
        reason=reason,
        start_time=start_time,
        rollup_time=rollup_time,
        resource=resource,
        rate=rate,
        quantity=quantity,
    )


def test_retried_attempts_cost_sums_quantity_rate_and_duration_over_resources_and_attempts():
    rows = [
        _attempt_resource('a', 'preempted', start_time=0, rollup_time=1000, rate=0.5, quantity=2),
        _attempt_resource(
            'a', 'preempted', start_time=0, rollup_time=1000, resource='service-fee/1', rate=0.25, quantity=4
        ),
        _attempt_resource('b', 'does_not_exist', start_time=100, rollup_time=200, rate=1.0, quantity=3),
        _attempt_resource('c', 'completed'),
    ]
    assert retried_attempts_cost(rows, outcome_attempt_id='c') == 1000 + 1000 + 300


def test_retried_attempts_cost_excludes_the_outcome_attempt_and_cancelled_attempts():
    rows = [
        _attempt_resource('a', 'cancelled'),
        _attempt_resource('b', 'completed'),
    ]
    assert retried_attempts_cost(rows, outcome_attempt_id='b') == 0.0


def test_retried_attempts_cost_counts_every_other_reason():
    reasons = [
        'preempted',
        'terminated',
        'does_not_exist',
        'deleted',
        'not_responding',
        'deactivated',
        'a-future-reason',
    ]
    rows = [_attempt_resource(str(i), reason, quantity=1, rate=1.0) for i, reason in enumerate(reasons)]
    assert retried_attempts_cost(rows, outcome_attempt_id='outcome') == 1000 * len(reasons)


def test_retried_attempts_cost_of_job_cancelled_from_ready_counts_earlier_attempts():
    rows = [
        _attempt_resource('a', 'preempted', quantity=1, rate=1.0),
        _attempt_resource('b', 'cancelled', quantity=1, rate=1.0),
    ]
    assert retried_attempts_cost(rows, outcome_attempt_id=None) == 1000


def test_retried_attempts_cost_of_attempts_without_duration_is_zero():
    rows = [
        _attempt_resource('never-activated', 'does_not_exist', start_time=None, rollup_time=1000),
        _attempt_resource('never-rolled-up', 'preempted', start_time=0, rollup_time=None),
        _attempt_resource('clock-skew', 'preempted', start_time=1000, rollup_time=500),
    ]
    assert retried_attempts_cost(rows, outcome_attempt_id=None) == 0.0


_OUTCOME_ON_PREEMPTIBLE = [
    _attempt_resource(
        'retried', 'preempted', start_time=0, rollup_time=1000, resource='compute/n1-preemptible/us-central1/500'
    ),
    _attempt_resource(
        'outcome',
        'completed',
        start_time=2000,
        rollup_time=3000,
        resource='compute/n1-preemptible/us-central1/500',
        rate=0.1,
        quantity=4,
    ),
    _attempt_resource(
        'outcome',
        'completed',
        start_time=2000,
        rollup_time=3000,
        resource='disk/local-ssd/preemptible/us-central1/500',
        rate=0.01,
        quantity=10,
    ),
    _attempt_resource(
        'outcome',
        'completed',
        start_time=2000,
        rollup_time=3000,
        resource='disk/pd-ssd/us-central1/500',
        rate=0.02,
        quantity=5,
    ),
]

_NONPREEMPTIBLE_RATES = [
    ('compute/n1-nonpreemptible/us-central1/1000', 0.3),
    ('compute/n1-nonpreemptible/us-central1/2500', 99.0),
    ('disk/local-ssd/nonpreemptible/us-central1/1', 0.04),
]


def test_nonpreemptible_counterparts_are_those_of_the_outcome_attempt():
    rows = [
        *_OUTCOME_ON_PREEMPTIBLE,
        _attempt_resource('retried', 'preempted', resource='memory/n1-preemptible/us-central1/1'),
    ]
    assert nonpreemptible_counterparts(rows, 'outcome') == {
        'compute/n1-nonpreemptible/us-central1',
        'disk/local-ssd/nonpreemptible/us-central1',
    }
    assert nonpreemptible_counterparts(rows, None) == set()


def test_projected_nonpreemptible_cost_swaps_preemptible_rates_for_those_in_effect_at_outcome_start():
    projected = projected_nonpreemptible_cost(_OUTCOME_ON_PREEMPTIBLE, 'outcome', _NONPREEMPTIBLE_RATES)
    # 1000ms each: compute 4 * 0.3, local SSD 10 * 0.04, persistent disk at its billed 5 * 0.02
    assert projected == pytest.approx(1000 * (1.2 + 0.4 + 0.1))


def test_projected_nonpreemptible_cost_is_none_when_a_counterpart_rate_is_missing():
    rates = [('compute/n1-nonpreemptible/us-central1/1000', 0.3)]
    assert projected_nonpreemptible_cost(_OUTCOME_ON_PREEMPTIBLE, 'outcome', rates) is None


def test_projected_nonpreemptible_cost_is_none_when_no_counterpart_rate_was_yet_in_effect():
    rates = [('compute/n1-nonpreemptible/us-central1/2500', 0.3), ('disk/local-ssd/nonpreemptible/us-central1/1', 0.04)]
    assert projected_nonpreemptible_cost(_OUTCOME_ON_PREEMPTIBLE, 'outcome', rates) is None


def test_projected_nonpreemptible_cost_is_none_when_the_outcome_attempt_was_not_preemptible():
    rows = [_attempt_resource('outcome', 'completed', resource='compute/n1-nonpreemptible/us-central1/1')]
    assert projected_nonpreemptible_cost(rows, 'outcome', _NONPREEMPTIBLE_RATES) is None


def test_projected_nonpreemptible_cost_is_none_without_an_outcome_attempt():
    assert projected_nonpreemptible_cost(_OUTCOME_ON_PREEMPTIBLE, None, _NONPREEMPTIBLE_RATES) is None
