from typing import List, Dict, Any

from ...instance_config import InstanceConfig, is_power_two
from .resource_utils import gcp_machine_type_to_dict


GCP_INSTANCE_CONFIG_VERSION = 4


class GCPSlimInstanceConfig(InstanceConfig):
    def __init__(self,
                 machine_type: str,
                 preemptible: bool,
                 local_ssd_data_disk: bool,
                 data_disk_size_gb: int,
                 boot_disk_size_gb: int,
                 job_private: bool,
                 ):
        self.cloud = 'gcp'
        self._machine_type = machine_type
        self.preemptible = preemptible
        self.local_ssd_data_disk = local_ssd_data_disk
        self.data_disk_size_gb = data_disk_size_gb
        self.job_private = job_private
        self.boot_disk_size_gb = boot_disk_size_gb

        machine_type_parts = gcp_machine_type_to_dict(self._machine_type)
        assert machine_type_parts is not None, machine_type
        self._instance_family = machine_type_parts['machine_family']
        self.worker_type = machine_type_parts['worker_type']
        self.cores = machine_type_parts['cores']

    @staticmethod
    def from_dict(data: dict) -> 'GCPSlimInstanceConfig':
        if data['version'] < 4:
            disks = data['disks']
            assert len(disks) == 2
            assert disks[0]['boot']
            boot_disk_size_gb = disks[0]['size']
            assert not disks[1]['boot']
            local_ssd_data_disk = disks[1]['type'] == 'local-ssd'
            data_disk_size_gb = disks[1]['size']
            job_private = data['job-private']
            return GCPSlimInstanceConfig(
                data['vm_config']['machineType'],
                data['instance']['preemptible'],
                local_ssd_data_disk,
                data_disk_size_gb,
                boot_disk_size_gb,
                job_private,
            )
        return GCPSlimInstanceConfig(
            data['machine_type'],
            data['preemptible'],
            data['local_ssd_data_disk'],
            data['data_disk_size_gb'],
            data['boot_disk_size_gb'],
            data['job_private'],
        )

    def to_dict(self) -> dict:
        return {
            'version': GCP_INSTANCE_CONFIG_VERSION,
            'cloud': 'gcp',
            'machine_type': self.machine_type,
            'preemptible': self.preemptible,
            'local_ssd_data_disk': self.local_ssd_data_disk,
            'data_disk_size_gb': self.data_disk_size_gb,
            'boot_disk_size_gb': self.boot_disk_size_gb,
            'job_private': self.job_private
        }

    @property
    def machine_type(self) -> str:
        return self._machine_type

    def resources(self, cpu_in_mcpu: int, memory_in_bytes: int, extra_storage_in_gib: int) -> List[Dict[str, Any]]:
        assert memory_in_bytes % (1024 * 1024) == 0, memory_in_bytes
        assert isinstance(extra_storage_in_gib, int), extra_storage_in_gib

        resources = []

        preemptible = 'preemptible' if self.preemptible else 'nonpreemptible'
        worker_fraction_in_1024ths = 1024 * cpu_in_mcpu // (self.cores * 1000)

        resources.append({'name': f'compute/{self._instance_family}-{preemptible}/1', 'quantity': cpu_in_mcpu})

        resources.append(
            {'name': f'memory/{self._instance_family}-{preemptible}/1', 'quantity': memory_in_bytes // 1024 // 1024}
        )

        # storage is in units of MiB
        resources.append({'name': 'disk/pd-ssd/1', 'quantity': extra_storage_in_gib * 1024})

        if self.local_ssd_data_disk:
            data_disk_name = 'disk/local-ssd/1'
        else:
            data_disk_name = 'disk/pd-ssd/1'
        resources.append(
            # the factors of 1024 cancel between GiB -> MiB and fraction_1024 -> fraction
            {'name': data_disk_name, 'quantity': self.data_disk_size_gb * worker_fraction_in_1024ths})

        resources.append(
            {'name': 'dsik/pd-ssd/1', 'quantity': self.boot_disk_size_gb})

        resources.append({'name': 'service-fee/1', 'quantity': cpu_in_mcpu})

        assert is_power_two(self.cores) and self.cores <= 256, self.cores
        resources.append({'name': 'ip-fee/1024/1', 'quantity': worker_fraction_in_1024ths})

        return resources
