import re
import logging
from typing import Any, Dict, Optional, Tuple

from ...globals import RESERVED_STORAGE_GB_PER_CORE

log = logging.getLogger('utils')

GCP_MAX_PERSISTENT_SSD_SIZE_GIB = 64 * 1024
MACHINE_TYPE_REGEX = re.compile('(?P<machine_family>[^-]+)-(?P<machine_type>[^-]+)-(?P<cores>\\d+)')
GCP_MACHINE_FAMILY = 'n1'


gcp_valid_machine_types = []
for typ in ('highcpu', 'standard', 'highmem'):
    if typ == 'standard':
        possible_cores = [1, 2, 4, 8, 16, 32, 64, 96]
    else:
        possible_cores = [2, 4, 8, 16, 32, 64, 96]
    for cores in possible_cores:
        gcp_valid_machine_types.append(f'{GCP_MACHINE_FAMILY}-{typ}-{cores}')


gcp_memory_to_worker_type = {
    'lowmem': 'highcpu',
    'standard': 'standard',
    'highmem': 'highmem'
}


def gcp_machine_type_to_dict(machine_type: str) -> Optional[Dict[str, Any]]:
    match = MACHINE_TYPE_REGEX.fullmatch(machine_type)
    if match is None:
        return match
    return match.groupdict()


def gcp_machine_type_to_worker_type_cores(machine_type: str) -> Tuple[str, int]:
    # FIXME: "WORKER TYPE" IS WRONG OR CONFUSING WHEN THE MACHINE TYPE IS NOT n1!
    maybe_machine_type_dict = gcp_machine_type_to_dict(machine_type)
    if maybe_machine_type_dict is None:
        raise ValueError(f'bad machine_type: {machine_type}')
    worker_type = maybe_machine_type_dict['machine_type']
    cores = int(maybe_machine_type_dict['cores'])
    return (worker_type, cores)


def family_worker_type_cores_to_gcp_machine_type(family: str, worker_type: str, cores: int) -> str:
    return f'{family}-{worker_type}-{cores}'


def gcp_cost_from_msec_mcpu(msec_mcpu: int) -> float:
    assert msec_mcpu is not None

    worker_type = 'standard'
    worker_cores = 16
    worker_disk_size_gb = 100

    # https://cloud.google.com/compute/all-pricing

    # per instance costs
    # persistent SSD: $0.17 GB/month
    # average number of days per month = 365.25 / 12 = 30.4375
    avg_n_days_per_month = 30.4375

    disk_cost_per_instance_hour = 0.17 * worker_disk_size_gb / avg_n_days_per_month / 24

    ip_cost_per_instance_hour = 0.004

    instance_cost_per_instance_hour = disk_cost_per_instance_hour + ip_cost_per_instance_hour

    # per core costs
    if worker_type == 'standard':
        cpu_cost_per_core_hour = 0.01
    elif worker_type == 'highcpu':
        cpu_cost_per_core_hour = 0.0075
    else:
        assert worker_type == 'highmem'
        cpu_cost_per_core_hour = 0.0125

    service_cost_per_core_hour = 0.01

    total_cost_per_core_hour = (
        cpu_cost_per_core_hour + instance_cost_per_instance_hour / worker_cores + service_cost_per_core_hour
    )

    return (msec_mcpu * 0.001 * 0.001) * (total_cost_per_core_hour / 3600)


def gcp_worker_memory_per_core_mib(worker_type: str) -> int:
    if worker_type == 'standard':
        m = 3840
    elif worker_type == 'highmem':
        m = 6656
    else:
        assert worker_type == 'highcpu', worker_type
        m = 924  # this number must be divisible by 4. I rounded up to the nearest MiB
    return m


def gcp_total_worker_storage_gib(worker_local_ssd_data_disk: bool, worker_pd_ssd_data_disk_size_gib: int) -> int:
    reserved_image_size = 25
    if worker_local_ssd_data_disk:
        # local ssd is 375Gi
        # reserve 25Gi for images
        return 375 - reserved_image_size
    return worker_pd_ssd_data_disk_size_gib - reserved_image_size


def gcp_unreserved_worker_data_disk_size_gib(worker_local_ssd_data_disk: bool, worker_pd_ssd_data_disk_size_gib: int,
                                             worker_cores: int) -> int:
    reserved_image_size = 30
    reserved_container_size = RESERVED_STORAGE_GB_PER_CORE * worker_cores
    if worker_local_ssd_data_disk:
        # local ssd is 375Gi
        # reserve 20Gi for images
        return 375 - reserved_image_size - reserved_container_size
    return worker_pd_ssd_data_disk_size_gib - reserved_image_size - reserved_container_size


def gcp_requested_to_actual_storage_bytes(storage_bytes, allow_zero_storage):
    if storage_bytes > GCP_MAX_PERSISTENT_SSD_SIZE_GIB * 1024 ** 3:
        return None
    if allow_zero_storage and storage_bytes == 0:
        return storage_bytes
    # minimum storage for a GCE instance is 10Gi
    return max(10 * 1024**3, storage_bytes)


def gcp_is_valid_storage_request(storage_in_gib: int) -> bool:
    return 10 <= storage_in_gib <= GCP_MAX_PERSISTENT_SSD_SIZE_GIB
