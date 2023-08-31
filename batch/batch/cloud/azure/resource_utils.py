import logging
import re
from typing import Dict, Optional, Tuple

import sortedcontainers

from ...globals import RESERVED_STORAGE_GB_PER_CORE

log = logging.getLogger('resource_utils')

# https://docs.microsoft.com/en-us/azure/virtual-machines/vm-naming-conventions
MACHINE_TYPE_REGEX = re.compile(
    r'(?P<typ>[^_]+)_(?P<family>[A-Z])(?P<sub_family>[^\d])?(?P<cpu>\d+)(-(?P<constrained_cpu>\d+))?(?P<additive_features>[^_]+)?(_((?P<accelerator_type>[^_]+)_)?(?P<version>.*)?)?'
)

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


azure_valid_machine_types = []
for cores in azure_valid_cores_from_worker_type['D']:
    azure_valid_machine_types.append(f'Standard_D{cores}ds_v4')
    azure_valid_machine_types.append(f'Standard_D{cores}s_v4')
for cores in azure_valid_cores_from_worker_type['E']:
    azure_valid_machine_types.append(f'Standard_E{cores}ds_v4')
    azure_valid_machine_types.append(f'Standard_E{cores}s_v4')
for cores in azure_valid_cores_from_worker_type['F']:
    azure_valid_machine_types.append(f'Standard_F{cores}s_v2')


azure_memory_to_worker_type = {'lowmem': 'F', 'standard': 'D', 'highmem': 'E'}


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


class MachineTypeParts:
    @staticmethod
    def from_dict(data: dict) -> 'MachineTypeParts':
        constrained_cpu = data['constrained_cpu']
        if constrained_cpu is not None:
            constrained_cpu = int(constrained_cpu)
        return MachineTypeParts(
            data['typ'],
            data['family'],
            data['sub_family'],
            int(data['cpu']),
            constrained_cpu,
            data['additive_features'],
            data['accelerator_type'],
            data['version'],
        )

    def __init__(
        self,
        typ: str,
        family: str,
        sub_family: Optional[str],
        cores: int,
        constrained_cpu: Optional[int],
        additive_features: Optional[str],
        accelerator_type: Optional[str],
        version: Optional[str],
    ):
        self.typ = typ
        self.family = family
        self.sub_family = sub_family
        self.cores = cores
        self.constrained_cpu = constrained_cpu
        self.additive_features = additive_features
        self.accelerator_type = accelerator_type
        self.version = version


def azure_machine_type_to_parts(machine_type: str) -> Optional[MachineTypeParts]:
    match = MACHINE_TYPE_REGEX.fullmatch(machine_type)
    if match is None:
        return match
    return MachineTypeParts.from_dict(match.groupdict())


def azure_machine_type_to_worker_type_and_cores(machine_type: str) -> Tuple[str, int]:
    maybe_machine_type_parts = azure_machine_type_to_parts(machine_type)
    if maybe_machine_type_parts is None:
        raise ValueError(f'bad machine_type: {machine_type}')
    return (maybe_machine_type_parts.family, maybe_machine_type_parts.cores)


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
