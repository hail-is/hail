import logging
import math
from typing import Dict, List, Optional, Tuple

from ..globals import RESERVED_STORAGE_GB_PER_CORE
from .azure.resource_utils import (
    azure_is_valid_storage_request,
    azure_local_ssd_size,
    azure_machine_type_to_cores_and_memory_bytes,
    azure_memory_to_worker_type,
    azure_requested_to_actual_storage_bytes,
    azure_valid_cores_from_worker_type,
    azure_valid_machine_types,
)
from .gcp.resource_utils import (
    gcp_is_valid_storage_request,
    gcp_local_ssd_size,
    gcp_machine_type_to_cores_and_memory_bytes,
    gcp_memory_to_worker_type,
    gcp_requested_to_actual_storage_bytes,
    gcp_valid_cores_from_worker_type,
    gcp_valid_machine_types,
)

log = logging.getLogger('resource_utils')


def round_up_division(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def possible_cores_from_worker_type(cloud: str, worker_type: str) -> List[int]:
    if cloud == 'azure':
        return azure_valid_cores_from_worker_type[worker_type]
    assert cloud == 'gcp'
    return gcp_valid_cores_from_worker_type[worker_type]


def valid_machine_types(cloud: str) -> List[str]:
    if cloud == 'azure':
        return azure_valid_machine_types
    assert cloud == 'gcp'
    return gcp_valid_machine_types


def memory_to_worker_type(cloud: str) -> Dict[str, str]:
    if cloud == 'azure':
        return azure_memory_to_worker_type
    assert cloud == 'gcp'
    return gcp_memory_to_worker_type


def machine_type_to_cores_and_memory_bytes(cloud: str, machine_type: str) -> Tuple[int, int]:
    if cloud == 'azure':
        return azure_machine_type_to_cores_and_memory_bytes(machine_type)
    assert cloud == 'gcp'
    return gcp_machine_type_to_cores_and_memory_bytes(machine_type)


def unreserved_worker_data_disk_size_gib(data_disk_size_gib: int, cores: int) -> int:
    reserved_image_size = 30
    reserved_container_size = RESERVED_STORAGE_GB_PER_CORE * cores
    return data_disk_size_gib - reserved_image_size - reserved_container_size


def requested_storage_bytes_to_actual_storage_gib(
    cloud: str, storage_bytes: int, allow_zero_storage: bool
) -> Optional[int]:
    if cloud == 'azure':
        actual_storage_bytes = azure_requested_to_actual_storage_bytes(storage_bytes, allow_zero_storage)
    else:
        assert cloud == 'gcp'
        actual_storage_bytes = gcp_requested_to_actual_storage_bytes(storage_bytes, allow_zero_storage)

    if actual_storage_bytes is None:
        return None

    return round_storage_bytes_to_gib(actual_storage_bytes)


def adjust_cores_for_packability(cores_in_mcpu: int) -> int:
    cores_in_mcpu = max(1, cores_in_mcpu)
    power = max(-2, math.ceil(math.log2(cores_in_mcpu / 1000)))
    return int(2**power * 1000)


def round_storage_bytes_to_gib(storage_bytes: int) -> int:
    gib = storage_bytes / 1024 / 1024 / 1024
    gib = math.ceil(gib)
    return gib


def storage_gib_to_bytes(storage_gib: int) -> int:
    return math.ceil(storage_gib * 1024**3)


def is_valid_cores_mcpu(cores_mcpu: int) -> bool:
    if cores_mcpu <= 0:
        return False
    quarter_core_mcpu = cores_mcpu * 4
    if quarter_core_mcpu % 1000 != 0:
        return False
    quarter_cores = quarter_core_mcpu // 1000
    return quarter_cores & (quarter_cores - 1) == 0


def is_valid_storage_request(cloud: str, storage_in_gib: int) -> bool:
    if cloud == 'azure':
        return azure_is_valid_storage_request(storage_in_gib)
    assert cloud == 'gcp'
    return gcp_is_valid_storage_request(storage_in_gib)


def local_ssd_size(cloud: str, worker_type: str, cores: int) -> int:
    if cloud == 'azure':
        return azure_local_ssd_size(worker_type, cores)
    assert cloud == 'gcp', cloud
    return gcp_local_ssd_size()
