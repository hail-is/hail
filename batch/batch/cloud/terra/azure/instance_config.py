import uuid
from typing import List

from hailtop.utils import filter_none

from ....driver.billing_manager import ProductVersions
from ...azure.instance_config import AzureSlimInstanceConfig
from ...azure.resources import AzureResource, AzureStaticSizedDiskResource, AzureVMResource, azure_resource_from_dict


class TerraAzureSlimInstanceConfig(AzureSlimInstanceConfig):
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
    ) -> 'TerraAzureSlimInstanceConfig':
        resource_id = str(uuid.uuid4())
        disk_resource_id = str(uuid.uuid4())

        resources: List[AzureResource] = filter_none([
            AzureVMResource.create(product_versions, machine_type, preemptible, location),
            AzureStaticSizedDiskResource.create(product_versions, 'E', boot_disk_size_gb, location),
        ])

        return TerraAzureSlimInstanceConfig(
            machine_type=machine_type,
            preemptible=preemptible,
            local_ssd_data_disk=local_ssd_data_disk,
            data_disk_size_gb=data_disk_size_gb,
            boot_disk_size_gb=boot_disk_size_gb,
            job_private=job_private,
            resources=resources,
            resource_id=resource_id,
            disk_resource_id=disk_resource_id,
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
        resource_id: str,
        disk_resource_id: str,
    ):
        super().__init__(
            machine_type=machine_type,
            preemptible=preemptible,
            local_ssd_data_disk=local_ssd_data_disk,
            data_disk_size_gb=data_disk_size_gb,
            boot_disk_size_gb=boot_disk_size_gb,
            job_private=job_private,
            resources=resources,
        )
        self._resource_id = resource_id
        self._disk_resource_id = disk_resource_id

    @staticmethod
    def from_dict(data: dict) -> 'TerraAzureSlimInstanceConfig':
        resources = data.get('resources', [])
        resources = [azure_resource_from_dict(resource) for resource in resources]
        return TerraAzureSlimInstanceConfig(
            data['machine_type'],
            data['preemptible'],
            data['local_ssd_data_disk'],
            data['data_disk_size_gb'],
            data['boot_disk_size_gb'],
            data['job_private'],
            resources,
            data['resource_id'],
            data['disk_resource_id'],
        )

    def to_dict(self) -> dict:
        azure_dict = super().to_dict()
        azure_dict.update({
            'resource_id': self._resource_id,
            'disk_resource_id': self._disk_resource_id,
        })
        return azure_dict
