from typing import List

from hailtop.utils import filter_none

from ...driver.billing_manager import ProductVersions
from ...instance_config import InstanceConfig
from .resource_utils import azure_machine_type_to_worker_type_and_cores, azure_worker_memory_per_core_mib
from .resources import (
    AzureDynamicSizedDiskResource,
    AzureIPFeeResource,
    AzureResource,
    AzureServiceFeeResource,
    AzureStaticSizedDiskResource,
    AzureVMResource,
    azure_resource_from_dict,
)

AZURE_INSTANCE_CONFIG_VERSION = 2


class AzureSlimInstanceConfig(InstanceConfig):
    @staticmethod
    def create(
        product_versions: ProductVersions,
        machine_type: str,
        preemptible: bool,
        local_ssd_data_disk: bool,
        data_disk_size_gb: int,
        boot_disk_size_gb: int,
        job_private: bool,
        location: str,
    ) -> 'AzureSlimInstanceConfig':
        if local_ssd_data_disk:
            data_disk_resource = None
        else:
            data_disk_resource = AzureStaticSizedDiskResource.create(product_versions, 'P', data_disk_size_gb, location)

        resources: List[AzureResource] = filter_none(
            [
                AzureVMResource.create(product_versions, machine_type, preemptible, location),
                AzureStaticSizedDiskResource.create(product_versions, 'E', boot_disk_size_gb, location),
                data_disk_resource,
                AzureDynamicSizedDiskResource.create(product_versions, 'P', location),
                AzureIPFeeResource.create(product_versions, 1024),
                AzureServiceFeeResource.create(product_versions),
            ]
        )

        return AzureSlimInstanceConfig(
            machine_type=machine_type,
            preemptible=preemptible,
            local_ssd_data_disk=local_ssd_data_disk,
            data_disk_size_gb=data_disk_size_gb,
            boot_disk_size_gb=boot_disk_size_gb,
            job_private=job_private,
            resources=resources,
        )

    def __init__(
        self,
        machine_type: str,
        preemptible: bool,
        local_ssd_data_disk: bool,
        data_disk_size_gb: int,
        boot_disk_size_gb: int,
        job_private: bool,
        resources: List[AzureResource],
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

    def total_memory_mib(self) -> int:
        return azure_worker_memory_per_core_mib(self._worker_type) * self.cores

    def worker_type(self) -> str:
        return self._worker_type

    def region_for(self, location: str) -> str:
        return location

    @staticmethod
    def from_dict(data: dict) -> 'AzureSlimInstanceConfig':
        resources = data.get('resources')
        if resources is None:
            assert data['version'] == 1, data
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
            'resources': [resource.to_dict() for resource in self.resources],
        }
