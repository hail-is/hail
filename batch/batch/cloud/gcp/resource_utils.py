import logging
import re
from typing import Optional, Tuple

log = logging.getLogger('utils')

GCP_MAX_PERSISTENT_SSD_SIZE_GIB = 64 * 1024
MACHINE_TYPE_REGEX = re.compile('(?P<machine_family>[^-]+)-(?P<machine_type>[^-]+)-(?P<cores>\\d+)')
GCP_MACHINE_FAMILY = 'n1'

MACHINE_FAMILY_TO_ACCELERATOR_VERSIONS = {"g2": "l4"}

gcp_valid_cores_from_worker_type = {
    'highcpu': [2, 4, 8, 16, 32, 64, 96],
    'standard': [1, 2, 4, 8, 16, 32, 64, 96],
    'highmem': [2, 4, 8, 16, 32, 64, 96],
}


gcp_valid_machine_types = []
gcp_valid_machine_types.append("g2-standard-4")
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


def gcp_machine_type_to_worker_type_and_cores(machine_type: str) -> Tuple[str, int]:
    # FIXME: "WORKER TYPE" IS WRONG OR CONFUSING WHEN THE MACHINE TYPE IS NOT n1!
    maybe_machine_type_parts = gcp_machine_type_to_parts(machine_type)
    if maybe_machine_type_parts is None:
        raise ValueError(f'bad machine_type: {machine_type}')
    return (maybe_machine_type_parts.worker_type, maybe_machine_type_parts.cores)


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
