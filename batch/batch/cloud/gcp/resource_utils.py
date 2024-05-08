import logging
import math
from typing import Optional, Tuple

log = logging.getLogger('utils')

GCP_MAX_PERSISTENT_SSD_SIZE_GIB = 64 * 1024

GCP_MACHINE_FAMILY = 'n1'

MEMORY_PER_CORE_MIB = {
    ('n1', 'standard'): 3840,
    ('n1', 'highmem'): 6656,
    ('n1', 'highcpu'): 924,
}


class GPUConfig:
    def __init__(self, num_gpus: int, gpu_type: str):
        self.num_gpus = num_gpus
        self.gpu_type = gpu_type


class MachineTypeParts:
    def __init__(self, machine_family: str, worker_type: str, cores: int, memory: int, gpu_config: Optional[GPUConfig]):
        self.machine_family = machine_family
        self.worker_type = worker_type
        self.cores = cores
        self.memory = memory
        self.gpu_config = gpu_config


MACHINE_TYPE_TO_PARTS = {
    'n1-standard-1': MachineTypeParts(
        cores=1,
        memory=int(3.75 * 1024**3),
        gpu_config=None,
        machine_family='n1',
        worker_type='standard',
    ),
    'n1-standard-2': MachineTypeParts(
        cores=2,
        memory=int(7.5 * 1024**3),
        gpu_config=None,
        machine_family='n1',
        worker_type='standard',
    ),
    'n1-standard-4': MachineTypeParts(
        cores=4,
        memory=int(15 * 1024**3),
        gpu_config=None,
        machine_family='n1',
        worker_type='standard',
    ),
    'n1-standard-8': MachineTypeParts(
        cores=8,
        memory=int(30 * 1024**3),
        gpu_config=None,
        machine_family='n1',
        worker_type='standard',
    ),
    'n1-standard-16': MachineTypeParts(
        cores=16,
        memory=int(60 * 1024**3),
        gpu_config=None,
        machine_family='n1',
        worker_type='standard',
    ),
    'n1-standard-32': MachineTypeParts(
        cores=32,
        memory=int(120 * 1024**3),
        gpu_config=None,
        machine_family='n1',
        worker_type='standard',
    ),
    'n1-standard-64': MachineTypeParts(
        cores=64,
        memory=int(240 * 1024**3),
        gpu_config=None,
        machine_family='n1',
        worker_type='standard',
    ),
    'n1-standard-96': MachineTypeParts(
        cores=96,
        memory=int(360 * 1024**3),
        gpu_config=None,
        machine_family='n1',
        worker_type='standard',
    ),
    'n1-highmem-2': MachineTypeParts(
        cores=2,
        memory=int(13 * 1024**3),
        gpu_config=None,
        machine_family='n1',
        worker_type='highmem',
    ),
    'n1-highmem-4': MachineTypeParts(
        cores=4,
        memory=int(26 * 1024**3),
        gpu_config=None,
        machine_family='n1',
        worker_type='highmem',
    ),
    'n1-highmem-8': MachineTypeParts(
        cores=8,
        memory=int(52 * 1024**3),
        gpu_config=None,
        machine_family='n1',
        worker_type='highmem',
    ),
    'n1-highmem-16': MachineTypeParts(
        cores=16,
        memory=int(104 * 1024**3),
        gpu_config=None,
        machine_family='n1',
        worker_type='highmem',
    ),
    'n1-highmem-32': MachineTypeParts(
        cores=32,
        memory=int(208 * 1024**3),
        gpu_config=None,
        machine_family='n1',
        worker_type='highmem',
    ),
    'n1-highmem-64': MachineTypeParts(
        cores=64,
        memory=int(416 * 1024**3),
        gpu_config=None,
        machine_family='n1',
        worker_type='highmem',
    ),
    'n1-highmem-96': MachineTypeParts(
        cores=96,
        memory=int(624 * 1024**3),
        gpu_config=None,
        machine_family='n1',
        worker_type='highmem',
    ),
    'n1-highcpu-2': MachineTypeParts(
        cores=2,
        # memory = 1.8 * 1024**3,
        memory=int(924 * 1024**2 * 2),
        gpu_config=None,
        machine_family='n1',
        worker_type='highcpu',
    ),
    'n1-highcpu-4': MachineTypeParts(
        cores=4,
        # memory = 3.6 * 1024**3,
        memory=int(924 * 1024**2 * 4),
        gpu_config=None,
        machine_family='n1',
        worker_type='highcpu',
    ),
    'n1-highcpu-8': MachineTypeParts(
        cores=8,
        # memory = 7.2 * 1024**3,
        memory=int(924 * 1024**2 * 8),
        gpu_config=None,
        machine_family='n1',
        worker_type='highcpu',
    ),
    'n1-highcpu-16': MachineTypeParts(
        cores=16,
        # memory = 14.4 * 1024**3,
        memory=int(924 * 1024**2 * 16),
        gpu_config=None,
        machine_family='n1',
        worker_type='highcpu',
    ),
    'n1-highcpu-32': MachineTypeParts(
        cores=32,
        # memory = 28.8 * 1024**3,
        memory=int(924 * 1024**2 * 32),
        gpu_config=None,
        machine_family='n1',
        worker_type='highcpu',
    ),
    'n1-highcpu-64': MachineTypeParts(
        cores=64,
        # memory = 57.6 * 1024**3,
        memory=int(924 * 1024**2 * 64),
        gpu_config=None,
        machine_family='n1',
        worker_type='highcpu',
    ),
    'n1-highcpu-96': MachineTypeParts(
        cores=96,
        # memory = 86.4 * 1024**3,
        memory=int(924 * 1024**2 * 96),
        gpu_config=None,
        machine_family='n1',
        worker_type='highcpu',
    ),
    'g2-standard-4': MachineTypeParts(
        cores=4,
        memory=int(16 * 1024**3),
        gpu_config=GPUConfig(num_gpus=1, gpu_type='l4'),
        machine_family='g2',
        worker_type='standard',
    ),
    'g2-standard-8': MachineTypeParts(
        cores=8,
        memory=int(32 * 1024**3),
        gpu_config=GPUConfig(num_gpus=1, gpu_type='l4'),
        machine_family='g2',
        worker_type='standard',
    ),
    'g2-standard-12': MachineTypeParts(
        cores=12,
        memory=int(48 * 1024**3),
        gpu_config=GPUConfig(num_gpus=1, gpu_type='l4'),
        machine_family='g2',
        worker_type='standard',
    ),
    'g2-standard-16': MachineTypeParts(
        cores=16,
        memory=int(64 * 1024**3),
        gpu_config=GPUConfig(num_gpus=1, gpu_type='l4'),
        machine_family='g2',
        worker_type='standard',
    ),
    'g2-standard-24': MachineTypeParts(
        cores=24,
        memory=int(96 * 1024**3),
        gpu_config=GPUConfig(num_gpus=2, gpu_type='l4'),
        machine_family='g2',
        worker_type='standard',
    ),
    'g2-standard-32': MachineTypeParts(
        cores=32,
        memory=int(128 * 1024**3),
        gpu_config=GPUConfig(num_gpus=1, gpu_type='l4'),
        machine_family='g2',
        worker_type='standard',
    ),
    'g2-standard-48': MachineTypeParts(
        cores=48,
        memory=int(192 * 1024**3),
        gpu_config=GPUConfig(num_gpus=4, gpu_type='l4'),
        machine_family='g2',
        worker_type='standard',
    ),
    'g2-standard-96': MachineTypeParts(
        cores=96,
        memory=int(384 * 1024**3),
        gpu_config=GPUConfig(num_gpus=8, gpu_type='l4'),
        machine_family='g2',
        worker_type='standard',
    ),
}


gcp_valid_cores_from_worker_type = {
    'highcpu': [2, 4, 8, 16, 32, 64, 96],
    'standard': [1, 2, 4, 8, 16, 32, 64, 96],
    'highmem': [2, 4, 8, 16, 32, 64, 96],
}


gcp_valid_machine_types = list(MACHINE_TYPE_TO_PARTS.keys())

gcp_memory_to_worker_type = {'lowmem': 'highcpu', 'standard': 'standard', 'highmem': 'highmem'}


def gcp_machine_type_to_parts(machine_type: str) -> Optional[MachineTypeParts]:
    return MACHINE_TYPE_TO_PARTS.get(machine_type)


def gcp_machine_type_to_cores_and_memory_bytes(machine_type: str) -> Tuple[int, int]:
    maybe_machine_type_parts = MACHINE_TYPE_TO_PARTS.get(machine_type)
    if maybe_machine_type_parts is None:
        raise ValueError(f'bad machine_type: {machine_type}')
    cores = maybe_machine_type_parts.cores
    memory_bytes = maybe_machine_type_parts.memory
    return cores, int(memory_bytes)


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


def machine_type_to_gpu(machine_type: str) -> Optional[str]:
    machine_type_parts = MACHINE_TYPE_TO_PARTS.get(machine_type)
    if machine_type_parts is None:
        return None
    if machine_type_parts.gpu_config is None:
        return None
    return machine_type_parts.gpu_config.gpu_type


def is_gpu(machine_family: str) -> bool:
    return machine_type_to_gpu(machine_family) is not None


def machine_type_to_gpu_num(machine_type: str) -> int:
    assert machine_type in MACHINE_TYPE_TO_PARTS
    machine_type_parts = MACHINE_TYPE_TO_PARTS[machine_type]
    if machine_type_parts.gpu_config is None:
        return 0
    return machine_type_parts.gpu_config.num_gpus


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
