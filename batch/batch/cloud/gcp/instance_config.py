from typing import List, Union

from ...driver.billing_manager import ProductVersions
from ...instance_config import InstanceConfig
from .resource_utils import gcp_machine_type_to_parts, machine_family_to_gpu
from .resources import (
    GCPAcceleratorResource,
    GCPComputeResource,
    GCPDynamicSizedDiskResource,
    GCPIPFeeResource,
    GCPLocalSSDStaticSizedDiskResource,
    GCPMemoryResource,
    GCPResource,
    GCPServiceFeeResource,
    GCPStaticSizedDiskResource,
    GCPSupportLogsSpecsAndFirewallFees,
    gcp_resource_from_dict,
)

GCP_INSTANCE_CONFIG_VERSION = 5


def region_from_location(location: str) -> str:
    return location.rsplit('-', maxsplit=1)[0]


class GCPSlimInstanceConfig(InstanceConfig):
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
    ) -> 'GCPSlimInstanceConfig':  # pylint: disable=unused-argument
        region = region_from_location(location)
        data_disk_resource: Union[GCPLocalSSDStaticSizedDiskResource, GCPStaticSizedDiskResource]
        if local_ssd_data_disk:
            data_disk_resource = GCPLocalSSDStaticSizedDiskResource.create(
                product_versions, data_disk_size_gb, preemptible, region
            )
        else:
            data_disk_resource = GCPStaticSizedDiskResource.create(
                product_versions, 'pd-ssd', data_disk_size_gb, region
            )

        machine_type_parts = gcp_machine_type_to_parts(machine_type)
        assert machine_type_parts is not None, machine_type
        instance_family = machine_type_parts.machine_family

        resources = [
            GCPComputeResource.create(product_versions, instance_family, preemptible, region),
            GCPMemoryResource.create(product_versions, instance_family, preemptible, region),
            GCPStaticSizedDiskResource.create(product_versions, 'pd-ssd', boot_disk_size_gb, region),
            data_disk_resource,
            GCPDynamicSizedDiskResource.create(product_versions, 'pd-ssd', region),
            GCPIPFeeResource.create(product_versions, 1024, preemptible),
            GCPServiceFeeResource.create(product_versions),
            GCPSupportLogsSpecsAndFirewallFees.create(product_versions),
        ]

        accelerator_family = machine_family_to_gpu(machine_type_parts.machine_family)
        if accelerator_family:
            resources.append(GCPAcceleratorResource.create(product_versions, accelerator_family, preemptible, region))

        return GCPSlimInstanceConfig(
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
        resources: List[GCPResource],
    ):
        self.cloud = 'gcp'
        self._machine_type = machine_type
        self.preemptible = preemptible
        self.local_ssd_data_disk = local_ssd_data_disk
        self.data_disk_size_gb = data_disk_size_gb
        self.job_private = job_private
        self.boot_disk_size_gb = boot_disk_size_gb

        machine_type_parts = gcp_machine_type_to_parts(self._machine_type)
        assert machine_type_parts is not None, machine_type
        self._instance_family = machine_type_parts.machine_family
        self._worker_type = machine_type_parts.worker_type
        self.cores = machine_type_parts.cores
        self.resources = resources

    def worker_type(self) -> str:
        return self._worker_type

    def region_for(self, location: str) -> str:
        # location = zone
        return region_from_location(location)

    @staticmethod
    def from_dict(data: dict) -> 'GCPSlimInstanceConfig':
        assert data['version'] == GCP_INSTANCE_CONFIG_VERSION

        machine_type = data['machine_type']
        preemptible = data['preemptible']
        local_ssd_data_disk = data['local_ssd_data_disk']
        data_disk_size_gb = data['data_disk_size_gb']
        boot_disk_size_gb = data['boot_disk_size_gb']
        job_private = data['job_private']

        assert 'resources' in data and data['resources'] is not None
        resources = [gcp_resource_from_dict(data) for data in data['resources']]

        return GCPSlimInstanceConfig(
            machine_type,
            preemptible,
            local_ssd_data_disk,
            data_disk_size_gb,
            boot_disk_size_gb,
            job_private,
            resources,
        )

    def to_dict(self) -> dict:
        return {
            'version': GCP_INSTANCE_CONFIG_VERSION,
            'cloud': 'gcp',
            'machine_type': self._machine_type,
            'preemptible': self.preemptible,
            'local_ssd_data_disk': self.local_ssd_data_disk,
            'data_disk_size_gb': self.data_disk_size_gb,
            'boot_disk_size_gb': self.boot_disk_size_gb,
            'job_private': self.job_private,
            'resources': [resource.to_dict() for resource in self.resources],
        }
