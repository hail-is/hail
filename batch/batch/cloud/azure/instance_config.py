from typing import List

from hailtop.utils import flatmap

from ...driver.billing_manager import ResourceVersions
from ...instance_config import InstanceConfig
from .resource_utils import azure_machine_type_to_worker_type_and_cores
from .resources import (AzureResource, AzureVMResource, AzureDiskResource, AzureExternalDiskResource,
                        AzureServiceFeeResource, AzureIPFeeResource, azure_resource_from_dict)


AZURE_INSTANCE_CONFIG_VERSION = 2


class AzureSlimInstanceConfig(InstanceConfig):
    @staticmethod
    def create(resource_versions: ResourceVersions,
               machine_type: str,
               preemptible: bool,
               local_ssd_data_disk: bool,
               data_disk_size_gb: int,
               boot_disk_size_gb: int,
               job_private: bool,
               location: str) -> 'AzureSlimInstanceConfig':
        if local_ssd_data_disk:
            data_disk_resource = None
        else:
            data_disk_resource = AzureDiskResource.new_resource(resource_versions, 'P', data_disk_size_gb, location)

        resources = flatmap([
            AzureVMResource.new_resource(resource_versions, machine_type, preemptible, location),
            AzureDiskResource.new_resource(resource_versions, 'P', boot_disk_size_gb, location),
            data_disk_resource,
            AzureExternalDiskResource.new_resource(resource_versions, 'P', location),
            AzureIPFeeResource.new_resource(resource_versions, 1024),
            AzureServiceFeeResource.new_resource(resource_versions),
        ])

        return AzureSlimInstanceConfig(
            machine_type=machine_type,
            preemptible=preemptible,
            local_ssd_data_disk=local_ssd_data_disk,
            data_disk_size_gb=data_disk_size_gb,
            boot_disk_size_gb=boot_disk_size_gb,
            job_private=job_private,
            resources=resources,
        )

    def __init__(self,
                 machine_type: str,
                 preemptible: bool,
                 local_ssd_data_disk: bool,
                 data_disk_size_gb: int,
                 boot_disk_size_gb: int,
                 job_private: bool,
                 resources: List[AzureResource]
                 ):
        self.cloud = 'azure'
        self._machine_type = machine_type
        self.preemptible = preemptible
        self.local_ssd_data_disk = local_ssd_data_disk
        self.data_disk_size_gb = data_disk_size_gb
        self.job_private = job_private
        self.boot_disk_size_gb = boot_disk_size_gb
        self.resources = resources

        worker_type, cores = azure_machine_type_to_worker_type_and_cores(self._machine_type)

        self._worker_type = worker_type
        self.cores = cores

    def worker_type(self) -> str:
        return self._worker_type

    @staticmethod
    def from_dict(data: dict) -> 'AzureSlimInstanceConfig':
        resources = data.get('resources')
        if resources is None:
            assert data['version'] == 1, data['version']
            resources = []
        resources = [azure_resource_from_dict(resource) for resource in resources]

        return AzureSlimInstanceConfig(
            data['machine_type'],
            data['preemptible'],
            data['local_ssd_data_disk'],
            data['data_disk_size_gb'],
            data['boot_disk_size_gb'],
            data['job_private'],
            resources,
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
            'job_private': self.job_private,
            'resources': [resource.to_dict() for resource in self.resources]
        }
