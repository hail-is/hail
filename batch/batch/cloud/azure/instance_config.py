from typing import List

from ...instance_config import InstanceConfig, QuantifiedResource
from .resource_utils import azure_machine_type_to_worker_type_and_cores


AZURE_INSTANCE_CONFIG_VERSION = 1


class AzureSlimInstanceConfig(InstanceConfig):
    def __init__(self,
                 machine_type: str,
                 preemptible: bool,
                 local_ssd_data_disk: bool,
                 data_disk_size_gb: int,
                 boot_disk_size_gb: int,
                 job_private: bool,
                 ):
        self.cloud = 'azure'
        self._machine_type = machine_type
        self.preemptible = preemptible
        self.local_ssd_data_disk = local_ssd_data_disk
        self.data_disk_size_gb = data_disk_size_gb
        self.job_private = job_private
        self.boot_disk_size_gb = boot_disk_size_gb

        worker_type, cores = azure_machine_type_to_worker_type_and_cores(self._machine_type)

        self._worker_type = worker_type
        self.cores = cores

    def worker_type(self) -> str:
        return self._worker_type

    @staticmethod
    def from_dict(data: dict) -> 'AzureSlimInstanceConfig':
        return AzureSlimInstanceConfig(
            data['machine_type'],
            data['preemptible'],
            data['local_ssd_data_disk'],
            data['data_disk_size_gb'],
            data['boot_disk_size_gb'],
            data['job_private'],
        )

    def to_dict(self) -> dict:
        return {
            'version': AZURE_INSTANCE_CONFIG_VERSION,
            'cloud': 'azure',
            'machine_type': self._machine_type,
            'preemptible': self.preemptible,
            'local_ssd_data_disk': self.local_ssd_data_disk,
            'data_disk_size_gb': self.data_disk_size_gb,
            'boot_disk_size_gb': self.boot_disk_size_gb,
            'job_private': self.job_private
        }

    def resources(self,
                  cpu_in_mcpu: int,
                  memory_in_bytes: int,
                  extra_storage_in_gib: int,
                  ) -> List[QuantifiedResource]:
        # TODO: Complete this method
        return []
