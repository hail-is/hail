from ..instance_config import InstanceConfig
from .resource_utils import (
    azure_machine_type_to_worker_type_cores_local_ssd,
    worker_properties_to_machine_type
)

from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from ..inst_coll_config import PoolConfig  # pylint: disable=cyclic-import

AZURE_INSTANCE_CONFIG_VERSION = 1


# azure_instance_config spec
#
# cloud: str
# version: int
# name: str
# machine_type: str
# cores: int
# job_private: bool
# vm_config: dict
# preemptible: bool
# worker_type: str
# location: str
# local_ssd_data_disk: bool
# data_disk_size_gb: int
# boot_disk_size_gb: int


class AzureInstanceConfig(InstanceConfig):
    @staticmethod
    def from_vm_config(vm_config: Dict[str, Any], job_private: bool = False):
        name = vm_config['properties']['parameters']['vmName']

        resources = vm_config['properties']['template']['resources']
        vm_resource = [resource for resource in resources
                       if resource['type'] == 'Microsoft.Compute/virtualMachines']
        assert len(vm_resource) == 1
        vm_resource = vm_resource[0]

        preemptible = vm_resource['properties']['priority'] == 'Spot'
        machine_type = vm_resource['properties']['hardwareProfile']['vmSize']

        worker_type, cores, local_ssd_data_disk = azure_machine_type_to_worker_type_cores_local_ssd(machine_type)

        if local_ssd_data_disk:
            data_disk_size_gb = 0
        else:
            data_disks = vm_resource['properties']['storageProfile']['dataDisks']
            assert len(data_disks) == 1
            data_disk = data_disks[0]
            data_disk_size_gb = data_disk['diskSizeGB']

        location = vm_resource['location']

        config = {
            'cloud': 'azure',
            'version': AZURE_INSTANCE_CONFIG_VERSION,
            'name': name,
            'machine_type': machine_type,
            'cores': cores,
            'job_private': job_private,
            'vm_config': vm_config,
            'preemptible': preemptible,
            'worker_type': worker_type,
            'location': location,
            'local_ssd_data_disk': local_ssd_data_disk,
            'data_disk_size_gb': data_disk_size_gb,
            'boot_disk_size_gb': 30,
        }

        return AzureInstanceConfig(config)

    @staticmethod
    def from_pool_config(pool_config: 'PoolConfig'):
        local_ssd_data_disk = pool_config.local_ssd_data_disk
        pd_ssd_data_disk_size_gb = pool_config.external_data_disk_size_gb
        worker_type = pool_config.worker_type
        cores = pool_config.worker_cores

        machine_type = worker_properties_to_machine_type(worker_type, cores, local_ssd_data_disk)

        config = {
            'cloud': 'azure',
            'version': AZURE_INSTANCE_CONFIG_VERSION,
            'name': None,
            'machine_type': machine_type,
            'cores': cores,
            'job_private': False,
            'vm_config': None,
            'spot': True,
            'worker_type': worker_type,
            'location': None,
            'local_ssd_data_disk': local_ssd_data_disk,
            'data_disk_size_gb': pd_ssd_data_disk_size_gb,
        }

        return AzureInstanceConfig(config)

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.cloud = self.config['cloud']
        assert self.cloud == 'azure'

        self.version = self.config['version']
        assert self.version >= 1

        self.name = self.config['name']
        self.machine_type = self.config['machine_type']
        self.cores = self.config['cores']
        self.job_private = self.config['job_private']
        self.vm_config = self.config['vm_config']
        self.spot = self.config['spot']
        self.worker_type = self.config['worker_type']
        self.machine_type = self.config['machine_type']
        self.location = self.config['location']
        self.local_ssd_data_disk = self.config['local_ssd_data_disk']
        self.data_disk_size_gb = self.config['data_disk_size_gb']

    def resources(self, cpu_in_mcpu: int, memory_in_bytes: int, storage_in_gib: int):
        return []
