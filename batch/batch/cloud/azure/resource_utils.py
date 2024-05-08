import logging
import math
from typing import Dict, Optional, Tuple

import sortedcontainers

from ...globals import RESERVED_STORAGE_GB_PER_CORE

log = logging.getLogger('resource_utils')

AZURE_MAX_PERSISTENT_SSD_SIZE_GIB = 32 * 1024


azure_local_ssd_size_per_core_by_worker_type = {
    'D': 37.5,
    'E': 37.5,
    'F': 8,
}


azure_valid_cores_from_worker_type = {
    'D': [2, 4, 8, 16, 32, 48, 64],
    'E': [2, 4, 8, 16, 20, 32, 48, 64],
    'F': [2, 4, 8, 16, 32, 48, 64, 72],
}

azure_memory_to_worker_type = {'lowmem': 'F', 'standard': 'D', 'highmem': 'E'}


class MachineTypeParts:
    def __init__(
        self,
        typ: str,
        family: str,
        cores: int,
        additive_features: Optional[str],
        version: Optional[str],
        memory: int,
    ):
        self.typ = typ
        self.family = family
        self.cores = cores
        self.additive_features = additive_features
        self.version = version
        self.memory = memory


MACHINE_TYPE_TO_PARTS = {
    'Standard_D2ds_v4': MachineTypeParts(
        typ='standard',
        family='D',
        cores=2,
        additive_features='ds',
        version='v4',
        memory=int(8 * 1024**3),
    ),
    'Standard_D4ds_v4': MachineTypeParts(
        typ='standard',
        family='D',
        cores=4,
        additive_features='ds',
        version='v4',
        memory=int(16 * 1024**3),
    ),
    'Standard_D8ds_v4': MachineTypeParts(
        typ='standard',
        family='D',
        cores=8,
        additive_features='ds',
        version='v4',
        memory=int(32 * 1024**3),
    ),
    'Standard_D16ds_v4': MachineTypeParts(
        typ='standard',
        family='D',
        cores=16,
        additive_features='ds',
        version='v4',
        memory=int(64 * 1024**3),
    ),
    'Standard_D32ds_v4': MachineTypeParts(
        typ='standard',
        family='D',
        cores=32,
        additive_features='ds',
        version='v4',
        memory=int(128 * 1024**3),
    ),
    'Standard_D48ds_v4': MachineTypeParts(
        typ='standard',
        family='D',
        cores=48,
        additive_features='ds',
        version='v4',
        memory=int(192 * 1024**3),
    ),
    'Standard_D64ds_v4': MachineTypeParts(
        typ='standard',
        family='D',
        cores=64,
        additive_features='ds',
        version='v4',
        memory=int(256 * 1024**3),
    ),
    'Standard_D2s_v4': MachineTypeParts(
        typ='standard',
        family='D',
        cores=2,
        additive_features='s',
        version='v4',
        memory=int(8 * 1024**3),
    ),
    'Standard_D4s_v4': MachineTypeParts(
        typ='standard',
        family='D',
        cores=4,
        additive_features='s',
        version='v4',
        memory=int(16 * 1024**3),
    ),
    'Standard_D8s_v4': MachineTypeParts(
        typ='standard',
        family='D',
        cores=8,
        additive_features='s',
        version='v4',
        memory=int(32 * 1024**3),
    ),
    'Standard_D16s_v4': MachineTypeParts(
        typ='standard',
        family='D',
        cores=16,
        additive_features='s',
        version='v4',
        memory=int(64 * 1024**3),
    ),
    'Standard_D32s_v4': MachineTypeParts(
        typ='standard',
        family='D',
        cores=32,
        additive_features='s',
        version='v4',
        memory=int(128 * 1024**3),
    ),
    'Standard_D48s_v4': MachineTypeParts(
        typ='standard',
        family='D',
        cores=48,
        additive_features='s',
        version='v4',
        memory=int(192 * 1024**3),
    ),
    'Standard_D64s_v4': MachineTypeParts(
        typ='standard',
        family='D',
        cores=64,
        additive_features='s',
        version='v4',
        memory=int(256 * 1024**3),
    ),
    'Standard_E2ds_v4': MachineTypeParts(
        typ='highmem',
        family='E',
        cores=2,
        additive_features='ds',
        version='v4',
        memory=int(16 * 1024**3),
    ),
    'Standard_E4ds_v4': MachineTypeParts(
        typ='highmem',
        family='E',
        cores=4,
        additive_features='ds',
        version='v4',
        memory=int(32 * 1024**3),
    ),
    'Standard_E8ds_v4': MachineTypeParts(
        typ='highmem',
        family='E',
        cores=8,
        additive_features='ds',
        version='v4',
        memory=int(64 * 1024**3),
    ),
    'Standard_E16ds_v4': MachineTypeParts(
        typ='highmem',
        family='E',
        cores=16,
        additive_features='ds',
        version='v4',
        memory=int(128 * 1024**3),
    ),
    'Standard_E20ds_v4': MachineTypeParts(
        typ='highmem',
        family='E',
        cores=20,
        additive_features='ds',
        version='v4',
        memory=int(160 * 1024**3),
    ),
    'Standard_E32ds_v4': MachineTypeParts(
        typ='highmem',
        family='E',
        cores=32,
        additive_features='ds',
        version='v4',
        memory=int(256 * 1024**3),
    ),
    'Standard_E48ds_v4': MachineTypeParts(
        typ='highmem',
        family='E',
        cores=48,
        additive_features='ds',
        version='v4',
        memory=int(384 * 1024**3),
    ),
    'Standard_E64ds_v4': MachineTypeParts(
        typ='highmem',
        family='E',
        cores=64,
        additive_features='ds',
        version='v4',
        memory=int(504 * 1024**3),
    ),
    'Standard_E2s_v4': MachineTypeParts(
        typ='highmem',
        family='E',
        cores=2,
        additive_features='s',
        version='v4',
        memory=int(16 * 1024**3),
    ),
    'Standard_E4s_v4': MachineTypeParts(
        typ='highmem',
        family='E',
        cores=4,
        additive_features='s',
        version='v4',
        memory=int(32 * 1024**3),
    ),
    'Standard_E8s_v4': MachineTypeParts(
        typ='highmem',
        family='E',
        cores=8,
        additive_features='s',
        version='v4',
        memory=int(64 * 1024**3),
    ),
    'Standard_E16s_v4': MachineTypeParts(
        typ='highmem',
        family='E',
        cores=16,
        additive_features='s',
        version='v4',
        memory=int(128 * 1024**3),
    ),
    'Standard_E20s_v4': MachineTypeParts(
        typ='highmem',
        family='E',
        cores=20,
        additive_features='s',
        version='v4',
        memory=int(160 * 1024**3),
    ),
    'Standard_E32s_v4': MachineTypeParts(
        typ='highmem',
        family='E',
        cores=32,
        additive_features='s',
        version='v4',
        memory=int(256 * 1024**3),
    ),
    'Standard_E48s_v4': MachineTypeParts(
        typ='highmem',
        family='E',
        cores=48,
        additive_features='s',
        version='v4',
        memory=int(384 * 1024**3),
    ),
    'Standard_E64s_v4': MachineTypeParts(
        typ='highmem',
        family='E',
        cores=64,
        additive_features='s',
        version='v4',
        memory=int(504 * 1024**3),
    ),
    'Standard_F2s_v2': MachineTypeParts(
        typ='lowmem',
        family='F',
        cores=2,
        additive_features='s',
        version='v2',
        memory=int(4 * 1024**3),
    ),
    'Standard_F4s_v2': MachineTypeParts(
        typ='lowmem',
        family='F',
        cores=4,
        additive_features='s',
        version='v2',
        memory=int(8 * 1024**3),
    ),
    'Standard_F8s_v2': MachineTypeParts(
        typ='lowmem',
        family='F',
        cores=8,
        additive_features='s',
        version='v2',
        memory=int(16 * 1024**3),
    ),
    'Standard_F16s_v2': MachineTypeParts(
        typ='lowmem',
        family='F',
        cores=16,
        additive_features='s',
        version='v2',
        memory=int(32 * 1024**3),
    ),
    'Standard_F32s_v2': MachineTypeParts(
        typ='lowmem',
        family='F',
        cores=32,
        additive_features='s',
        version='v2',
        memory=int(64 * 1024**3),
    ),
    'Standard_F48s_v2': MachineTypeParts(
        typ='lowmem',
        family='F',
        cores=48,
        additive_features='s',
        version='v2',
        memory=int(96 * 1024**3),
    ),
    'Standard_F64s_v2': MachineTypeParts(
        typ='lowmem',
        family='F',
        cores=64,
        additive_features='s',
        version='v2',
        memory=int(128 * 1024**3),
    ),
    'Standard_F72s_v2': MachineTypeParts(
        typ='lowmem',
        family='F',
        cores=72,
        additive_features='s',
        version='v2',
        memory=int(144 * 1024**3),
    ),
}

azure_valid_machine_types = list(MACHINE_TYPE_TO_PARTS.keys())


class AzureDisk:
    def __init__(self, name: str, size_in_gib: int):
        self.name = name
        self.size_in_gib = size_in_gib

    @property
    def family(self):
        family = self.name[0]
        assert family in azure_disk_families
        return family


# https://azure.microsoft.com/en-us/pricing/details/managed-disks/
azure_disk_families = {'P', 'E', 'S'}
azure_disk_number_to_storage_gib = {
    '1': 4,
    '2': 8,
    '3': 16,
    '4': 32,
    '6': 64,
    '10': 128,
    '15': 256,
    '20': 512,
    '30': 1024,
    '40': 2 * 1024,
    '50': 4 * 1024,
    '60': 8 * 1024,
    '70': 16 * 1024,
    '80': 32 * 1024,
}

azure_disks_by_disk_type = {
    'P': sortedcontainers.SortedSet(key=lambda disk: disk.size_in_gib),
    'E': sortedcontainers.SortedSet(key=lambda disk: disk.size_in_gib),
    'S': sortedcontainers.SortedSet(key=lambda disk: disk.size_in_gib),
}

valid_azure_disk_names = set()

azure_disk_name_to_storage_gib: Dict[str, int] = {}

for family in azure_disk_families:
    for disk_number, storage_gib in azure_disk_number_to_storage_gib.items():
        if family == 'S' and disk_number in ('1', '2', '3'):
            continue
        disk_name = f'{family}{disk_number}'
        azure_disks_by_disk_type[family].add(AzureDisk(disk_name, storage_gib))
        valid_azure_disk_names.add(disk_name)
        azure_disk_name_to_storage_gib[disk_name] = storage_gib


def azure_disk_from_storage_in_gib(disk_type: str, storage_in_gib: int) -> Optional[AzureDisk]:
    disks = azure_disks_by_disk_type[disk_type]
    index = disks.bisect_key_left(storage_in_gib)
    if index == len(disks):
        return None
    return disks[index]  # type: ignore


def azure_machine_type_to_parts(machine_type: str) -> Optional[MachineTypeParts]:
    return MACHINE_TYPE_TO_PARTS.get(machine_type)


def azure_machine_type_to_cores_and_memory_bytes(machine_type: str) -> Tuple[int, int]:
    maybe_machine_type_parts = azure_machine_type_to_parts(machine_type)
    if maybe_machine_type_parts is None:
        raise ValueError(f'bad machine_type: {machine_type}')
    cores = maybe_machine_type_parts.cores
    memory_bytes = maybe_machine_type_parts.memory
    return cores, memory_bytes


def azure_worker_properties_to_machine_type(worker_type: str, cores: int, local_ssd_data_disk: bool) -> str:
    if worker_type == 'F':
        return f'Standard_F{cores}s_v2'

    if local_ssd_data_disk:
        additive_features = 'ds'
    else:
        additive_features = 's'

    machine_type = f'Standard_{worker_type}{cores}{additive_features}_v4'
    return machine_type


def azure_worker_memory_per_core_mib(worker_type: str) -> int:
    if worker_type == 'F':
        m = 2048
    elif worker_type == 'D':
        m = 4096
    else:
        assert worker_type == 'E'
        m = 8192
    return m


def azure_local_ssd_size(worker_type: str, cores: int) -> int:
    return int(cores * azure_local_ssd_size_per_core_by_worker_type[worker_type])


def azure_unreserved_worker_data_disk_size_gib(
    local_ssd_data_disk, external_data_disk_size_gib, worker_cores, worker_type
):
    reserved_image_size = 30
    reserved_container_size = RESERVED_STORAGE_GB_PER_CORE * worker_cores
    if local_ssd_data_disk:
        return azure_local_ssd_size(worker_type, worker_cores) - reserved_image_size - reserved_container_size
    return external_data_disk_size_gib - reserved_image_size - reserved_container_size


def azure_requested_to_actual_storage_bytes(storage_bytes, allow_zero_storage):
    if storage_bytes > AZURE_MAX_PERSISTENT_SSD_SIZE_GIB * 1024**3:
        return None
    if allow_zero_storage and storage_bytes == 0:
        return storage_bytes
    # actual minimum storage size is 4 Gi on Azure, but keeping 10 to be consistent with gcp
    return max(10 * 1024**3, storage_bytes)


def azure_is_valid_storage_request(storage_in_gib: int) -> bool:
    return 10 <= storage_in_gib <= AZURE_MAX_PERSISTENT_SSD_SIZE_GIB


def azure_cores_mcpu_to_memory_bytes(mcpu: int, worker_type: str) -> int:
    memory_mib = azure_worker_memory_per_core_mib(worker_type)
    memory_bytes = int(memory_mib * 1024**2)
    return int((mcpu / 1000) * memory_bytes)


def azure_adjust_cores_for_memory_request(cores_in_mcpu: int, memory_in_bytes: int, worker_type: str) -> int:
    memory_per_core_mib = azure_worker_memory_per_core_mib(worker_type)
    memory_per_core_bytes = int(memory_per_core_mib * 1024**2)
    min_cores_mcpu = math.ceil((memory_in_bytes / memory_per_core_bytes) * 1000)
    return max(cores_in_mcpu, min_cores_mcpu)
