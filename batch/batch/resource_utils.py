import logging
import math
from typing import Tuple, Dict, List

from .gcp.resource_utils import (gcp_cost_from_msec_mcpu,
                                 gcp_worker_memory_per_core_mib,
                                 gcp_total_worker_storage_gib,
                                 gcp_unreserved_worker_data_disk_size_gib,
                                 gcp_requested_to_actual_storage_bytes,
                                 gcp_machine_type_to_worker_type_cores,
                                 gcp_valid_machine_types,
                                 gcp_memory_to_worker_type,
                                 gcp_is_valid_storage_request)

log = logging.getLogger('resource_utils')


def round_up_division(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def valid_machine_types(cloud: str) -> List[str]:
    assert cloud == 'gcp'
    return gcp_valid_machine_types


def memory_to_worker_type(cloud: str) -> Dict[str, str]:
    assert cloud == 'gcp'
    return gcp_memory_to_worker_type


def machine_type_to_worker_type_cores(cloud: str, machine_type: str) -> Tuple[str, int]:
    assert cloud == 'gcp'
    return gcp_machine_type_to_worker_type_cores(machine_type)


def cost_from_msec_mcpu(msec_mcpu):
    if msec_mcpu is None:
        return None
    # msec_mcpu is deprecated and only applicable to GCP
    return gcp_cost_from_msec_mcpu(msec_mcpu)


def worker_memory_per_core_mib(cloud, worker_type):
    assert cloud == 'gcp'
    return gcp_worker_memory_per_core_mib(worker_type)


def worker_memory_per_core_bytes(cloud, worker_type):
    m = worker_memory_per_core_mib(cloud, worker_type)
    return int(m * 1024 ** 2)


def memory_bytes_to_cores_mcpu(cloud, memory_in_bytes, worker_type):
    return math.ceil((memory_in_bytes / worker_memory_per_core_bytes(cloud, worker_type)) * 1000)


def cores_mcpu_to_memory_bytes(cloud, cores_in_mcpu, worker_type):
    return int((cores_in_mcpu / 1000) * worker_memory_per_core_bytes(cloud, worker_type))


def adjust_cores_for_memory_request(cloud, cores_in_mcpu, memory_in_bytes, worker_type):
    min_cores_mcpu = memory_bytes_to_cores_mcpu(cloud, memory_in_bytes, worker_type)
    return max(cores_in_mcpu, min_cores_mcpu)


def total_worker_storage_gib(cloud, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gib):
    assert cloud == 'gcp'
    return gcp_total_worker_storage_gib(worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gib)


def worker_storage_per_core_bytes(cloud, worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gib):
    return (
        total_worker_storage_gib(cloud, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gib) * 1024 ** 3
    ) // worker_cores


def storage_bytes_to_cores_mcpu(
    cloud, storage_in_bytes, worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gib
):
    return round_up_division(
        storage_in_bytes * 1000,
        worker_storage_per_core_bytes(cloud, worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gib),
    )


def adjust_cores_for_storage_request(
    cloud, cores_in_mcpu, storage_in_bytes, worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gib
):
    min_cores_mcpu = storage_bytes_to_cores_mcpu(
        cloud, storage_in_bytes, worker_cores, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gib
    )
    return max(cores_in_mcpu, min_cores_mcpu)


def unreserved_worker_data_disk_size_gib(cloud, worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gib, worker_cores):
    assert cloud == 'gcp'
    return gcp_unreserved_worker_data_disk_size_gib(worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gib, worker_cores)


def requested_storage_bytes_to_actual_storage_gib(cloud, storage_bytes, allow_zero_storage):
    assert cloud == 'gcp'
    actual_storage_bytes = gcp_requested_to_actual_storage_bytes(storage_bytes, allow_zero_storage)
    return round_storage_bytes_to_gib(actual_storage_bytes)


def adjust_cores_for_packability(cores_in_mcpu):
    cores_in_mcpu = max(1, cores_in_mcpu)
    power = max(-2, math.ceil(math.log2(cores_in_mcpu / 1000)))
    return int(2 ** power * 1000)


def round_storage_bytes_to_gib(storage_bytes):
    gib = storage_bytes / 1024 / 1024 / 1024
    gib = math.ceil(gib)
    return gib


def storage_gib_to_bytes(storage_gib):
    return math.ceil(storage_gib * 1024 ** 3)


def is_valid_cores_mcpu(cores_mcpu: int):
    if cores_mcpu <= 0:
        return False
    quarter_core_mcpu = cores_mcpu * 4
    if quarter_core_mcpu % 1000 != 0:
        return False
    quarter_cores = quarter_core_mcpu // 1000
    return quarter_cores & (quarter_cores - 1) == 0


def is_valid_storage_request(cloud, storage_in_gib: int) -> bool:
    assert cloud == 'gcp'
    return gcp_is_valid_storage_request(storage_in_gib)
