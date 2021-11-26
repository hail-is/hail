import abc
from typing import Dict, Any, Optional

from ...driver.billing_manager import ResourceVersions
from ...resources import (QuantifiedResource, Resource, DiskResourceMixin, VMResourceMixin, IPFeeResourceMixin,
                          ServiceFeeResourceMixin, ExternalDiskResourceMixin)
from .resource_utils import azure_disk_from_storage_in_gib, azure_disks_by_disk_type


class AzureResource(Resource, abc.ABC):
    pass


def azure_disk_prefix(disk_product: str, redundancy_type: str, location: str) -> str:
    return f'az/disk/{disk_product}_{redundancy_type}/{location}'


class AzureDiskResource(DiskResourceMixin, AzureResource):
    FORMAT_VERSION = 1
    TYPE = 'azure_disk'

    @staticmethod
    def generate_prefix(disk_product: str, redundancy_type: str, location: str) -> str:
        return azure_disk_prefix(disk_product, redundancy_type, location)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AzureDiskResource':
        assert data['type'] == AzureDiskResource.TYPE
        return AzureDiskResource(data['name'], data['storage_in_gib'])

    @staticmethod
    def new_resource(resource_versions: ResourceVersions,
                     disk_type: str,
                     storage_in_gib: int,
                     location: str,
                     redundancy_type: str = 'LRS',
                     ) -> 'AzureDiskResource':
        assert redundancy_type in ('LRS', 'ZRS'), redundancy_type

        # Azure bills for specific disk sizes so we must round the storage_in_gib to the nearest power of two
        disk = azure_disk_from_storage_in_gib(disk_type, storage_in_gib)
        assert disk, f'disk_type={disk_type} storage_in_gib={storage_in_gib}'

        prefix = AzureDiskResource.generate_prefix(disk.name, redundancy_type, location)
        name = resource_versions.latest_resource_name(prefix)
        return AzureDiskResource(name, disk.size_in_gib)

    def __init__(self, name: str, storage_in_gib: int):
        self.name = name
        self.storage_in_gib = storage_in_gib

    def to_dict(self) -> dict:
        return {
            'type': self.TYPE,
            'name': self.name,
            'storage_in_gib': self.storage_in_gib,
            'format_version': self.FORMAT_VERSION
        }


class AzureExternalDiskResource(ExternalDiskResourceMixin, AzureResource):
    FORMAT_VERSION = 1
    TYPE = 'azure_external_disk'

    @staticmethod
    def generate_prefix(disk_product: str, redundancy_type: str, location: str) -> str:
        return azure_disk_prefix(disk_product, redundancy_type, location)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AzureExternalDiskResource':
        assert data['type'] == AzureExternalDiskResource.TYPE
        return AzureExternalDiskResource(data['disk_type'], data['location'], data['latest_disk_versions'])

    @staticmethod
    def new_resource(resource_versions: ResourceVersions,
                     disk_type: str,
                     location: str,
                     redundancy_type: str = 'LRS',
                     ) -> 'AzureExternalDiskResource':
        assert redundancy_type in ('LRS', 'ZRS'), redundancy_type

        disk_name_to_resource_names = {}
        for disk in azure_disks_by_disk_type[disk_type]:
            disk_name = disk.name
            prefix = AzureExternalDiskResource.generate_prefix(disk_name, redundancy_type, location)
            resource_name = resource_versions.latest_resource_name(prefix)
            disk_name_to_resource_names[disk_name] = resource_name
        return AzureExternalDiskResource(disk_type, location, disk_name_to_resource_names)

    def __init__(self, disk_type: str, location: str, disk_name_to_resource_names: Dict[str, str]):
        self.disk_type = disk_type
        self.location = location
        self.disk_name_to_resource_names = disk_name_to_resource_names

    def to_quantified_resource(self,
                               cpu_in_mcpu: int,
                               memory_in_bytes: int,
                               worker_fraction_in_1024ths: int,
                               external_storage_in_gib: int) -> Optional[QuantifiedResource]:  # pylint: disable=unused-argument
        del cpu_in_mcpu, memory_in_bytes, worker_fraction_in_1024ths

        if external_storage_in_gib == 0:
            return None

        # Azure bills for specific disk sizes so we must round the storage_in_gib to the nearest power of two
        disk = azure_disk_from_storage_in_gib(self.disk_type, external_storage_in_gib)
        assert disk, f'disk_type={self.disk_type} storage_in_gib={external_storage_in_gib}'
        resource_name = self.disk_name_to_resource_names[disk.name]

        return {'name': resource_name, 'quantity': disk.size_in_gib * 1024}  # storage is in units of MiB

    def to_dict(self) -> dict:
        return {
            'type': self.TYPE,
            'disk_type': self.disk_type,
            'location': self.location,
            'latest_disk_versions': self.disk_name_to_resource_names,
            'format_version': self.FORMAT_VERSION
        }


class AzureVMResource(VMResourceMixin, AzureResource):
    FORMAT_VERSION = 1
    TYPE = 'azure_vm'

    @staticmethod
    def generate_prefix(machine_type: str, preemptible: bool, location: str) -> str:
        preemptible_str = 'spot' if preemptible else 'regular'
        return f'az/vm/{machine_type}/{preemptible_str}/{location}'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AzureVMResource':
        assert data['type'] == AzureVMResource.TYPE
        return AzureVMResource(data['name'])

    @staticmethod
    def new_resource(resource_versions: ResourceVersions,
                     machine_type: str,
                     preemptible: bool,
                     location: str,
                     ) -> 'AzureVMResource':
        prefix = AzureVMResource.generate_prefix(machine_type, preemptible, location)
        name = resource_versions.latest_resource_name(prefix)
        return AzureVMResource(name)

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict:
        return {
            'type': self.TYPE,
            'name': self.name,
            'format_version': self.FORMAT_VERSION
        }


class AzureServiceFeeResource(ServiceFeeResourceMixin, AzureResource):
    FORMAT_VERSION = 1
    TYPE = 'azure_service_fee'

    @staticmethod
    def generate_prefix():
        return 'az/service-fee'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AzureServiceFeeResource':
        assert data['type'] == AzureServiceFeeResource.TYPE
        return AzureServiceFeeResource(data['name'])

    @staticmethod
    def new_resource(resource_versions: ResourceVersions) -> 'AzureServiceFeeResource':
        prefix = AzureServiceFeeResource.generate_prefix()
        name = resource_versions.latest_resource_name(prefix)
        return AzureServiceFeeResource(name)

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict:
        return {
            'type': self.TYPE,
            'name': self.name,
            'format_version': self.FORMAT_VERSION
        }


class AzureIPFeeResource(IPFeeResourceMixin, AzureResource):
    FORMAT_VERSION = 1
    TYPE = 'azure_ip_fee'

    @staticmethod
    def generate_prefix(base: int):
        return f'az/ip-fee/{base}'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AzureIPFeeResource':
        assert data['type'] == AzureIPFeeResource.TYPE
        return AzureIPFeeResource(data['name'])

    @staticmethod
    def new_resource(resource_versions: ResourceVersions, base: int) -> 'AzureIPFeeResource':
        prefix = AzureIPFeeResource.generate_prefix(base)
        name = resource_versions.latest_resource_name(prefix)
        return AzureIPFeeResource(name)

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict:
        return {
            'type': self.TYPE,
            'name': self.name,
            'format_version': self.FORMAT_VERSION
        }


def azure_resource_from_dict(data: dict) -> AzureResource:
    typ = data['type']
    if typ == AzureDiskResource.TYPE:
        return AzureDiskResource.from_dict(data)
    if typ == AzureExternalDiskResource.TYPE:
        return AzureExternalDiskResource.from_dict(data)
    if typ == AzureVMResource.TYPE:
        return AzureVMResource.from_dict(data)
    if typ == AzureServiceFeeResource.TYPE:
        return AzureServiceFeeResource.from_dict(data)
    assert typ == AzureIPFeeResource.TYPE
    return AzureIPFeeResource.from_dict(data)
