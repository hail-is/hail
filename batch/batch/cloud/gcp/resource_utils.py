import logging
import math
import re
from typing import Optional, Tuple

log = logging.getLogger('utils')

GCP_MAX_PERSISTENT_SSD_SIZE_GIB = 64 * 1024
MACHINE_TYPE_REGEX = re.compile('(?P<machine_family>[^-]+)-(?P<machine_type>[^-]+)-(?P<cores>\\d+)')
GCP_MACHINE_FAMILY = 'n1'

MACHINE_FAMILY_TO_ACCELERATOR_VERSIONS = {'g2': 'l4'}

SINGLE_GPU_MACHINES = ['g2-standard-4','g2-standard-8','g2-standard-12','g2-standard-16','g2-standard-32']

TWO_GPU_MACHINES = ['g2-standard-24',]

FOUR_GPU_MACHINES = ['g2-standard-48',]

EIGHT_GPU_MACHINES = ['g2-standard-96',]


MEMORY_PER_CORE_MIB = {
    ('n1', 'standard'): 3840,
    ('n1', 'highmem'): 6656,
    ('n1', 'highcpu'): 924,
    ('g2', 'standard'): 4000,
}


gcp_valid_cores_from_worker_type = {
    'highcpu': [2, 4, 8, 16, 32, 64, 96],
    'standard': [1, 2, 4, 8, 16, 32, 64, 96],
    'highmem': [2, 4, 8, 16, 32, 64, 96],
}


gcp_valid_machine_types = SINGLE_GPU_MACHINES + TWO_GPU_MACHINES + FOUR_GPU_MACHINES + EIGHT_GPU_MACHINES
for typ in ('highcpu', 'standard', 'highmem'):
    possible_cores = gcp_valid_cores_from_worker_type[typ]
    for cores in possible_cores:
        gcp_valid_machine_types.append(f'{GCP_MACHINE_FAMILY}-{typ}-{cores}')


gcp_memory_to_worker_type = {'lowmem': 'highcpu', 'standard': 'standard', 'highmem': 'highmem'}


class MachineTypeParts:
    @staticmethod
    def from_dict(data: dict) -> 'MachineTypeParts':
        return MachineTypeParts(data['machine_family'], data['machine_type'], int(data['cores']))

    def __init__(self, machine_family: str, worker_type: str, cores: int):
        self.machine_family = machine_family
        self.worker_type = worker_type
        self.cores = cores


def gcp_machine_type_to_parts(machine_type: str) -> Optional[MachineTypeParts]:
    match = MACHINE_TYPE_REGEX.fullmatch(machine_type)
    if match is None:
        return match
    return MachineTypeParts.from_dict(match.groupdict())


def gcp_machine_type_to_cores_and_memory_mib_per_core(machine_type: str) -> Tuple[int, int]:
    # FIXME: "WORKER TYPE" IS WRONG OR CONFUSING WHEN THE MACHINE TYPE IS NOT n1!
    maybe_machine_type_parts = gcp_machine_type_to_parts(machine_type)
    if maybe_machine_type_parts is None:
        raise ValueError(f'bad machine_type: {machine_type}')
    cores = maybe_machine_type_parts.cores
    memory_per_core = gcp_worker_memory_per_core_mib(
        maybe_machine_type_parts.machine_family, maybe_machine_type_parts.worker_type
    )
    return cores, memory_per_core


def family_worker_type_cores_to_gcp_machine_type(family: str, worker_type: str, cores: int) -> str:
    return f'{family}-{worker_type}-{cores}'


def gcp_worker_memory_per_core_mib(machine_family: str, worker_type: str) -> int:
    machine_worker_key = (machine_family, worker_type)
    assert machine_worker_key in MEMORY_PER_CORE_MIB, machine_worker_key
    return MEMORY_PER_CORE_MIB[machine_worker_key]


def gcp_requested_to_actual_storage_bytes(storage_bytes, allow_zero_storage):
    if storage_bytes > GCP_MAX_PERSISTENT_SSD_SIZE_GIB * 1024**3:
        return None
    if allow_zero_storage and storage_bytes == 0:
        return storage_bytes
    # minimum storage for a GCE instance is 10Gi
    return max(10 * 1024**3, storage_bytes)


def gcp_is_valid_storage_request(storage_in_gib: int) -> bool:
    return 10 <= storage_in_gib <= GCP_MAX_PERSISTENT_SSD_SIZE_GIB


def gcp_local_ssd_size() -> int:
    return 375


def machine_family_to_gpu(machine_family: str) -> Optional[str]:
    return MACHINE_FAMILY_TO_ACCELERATOR_VERSIONS.get(machine_family)


def is_gpu(machine_family: str) -> bool:
    return machine_family_to_gpu(machine_family) is not None


def machine_type_to_gpu_num(machine_type: str) -> int:
    if machine_type in SINGLE_GPU_MACHINES:
        return 1
    elif machine_type in TWO_GPU_MACHINES:
        return 2
    elif machine_type in FOUR_GPU_MACHINES:
        return 4
    elif machine_type in EIGHT_GPU_MACHINES:
        return 8


def gcp_cores_mcpu_to_memory_bytes(mcpu: int, machine_family: str, worker_type: str) -> int:
    memory_mib = gcp_worker_memory_per_core_mib(machine_family, worker_type)
    memory_bytes = int(memory_mib * 1024**2)
    return int((mcpu / 1000) * memory_bytes)


def gcp_adjust_cores_for_memory_request(
    cores_in_mcpu: int, memory_in_bytes: int, machine_family: str, worker_type: str
) -> int:
    memory_per_core_mib = gcp_worker_memory_per_core_mib(machine_family, worker_type)
    memory_per_core_bytes = int(memory_per_core_mib * 1024**2)
    min_cores_mcpu = math.ceil((memory_in_bytes / memory_per_core_bytes) * 1000)
    return max(cores_in_mcpu, min_cores_mcpu)

