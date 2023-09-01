from typing import Any, Dict, Optional

from ...driver.billing_manager import ProductVersions
from ...resources import (
    DynamicSizedDiskResourceMixin,
    IPFeeResourceMixin,
    QuantifiedResource,
    Resource,
    ServiceFeeResourceMixin,
    StaticSizedDiskResourceMixin,
    VMResourceMixin,
)
from .resource_utils import azure_disk_from_storage_in_gib, azure_disks_by_disk_type


class AzureResource(Resource):
    pass


def azure_disk_product(disk_product: str, redundancy_type: str, location: str) -> str:
    return f'az/disk/{disk_product}_{redundancy_type}/{location}'


class AzureStaticSizedDiskResource(StaticSizedDiskResourceMixin, AzureResource):
    FORMAT_VERSION = 1
    TYPE = 'azure_static_sized_disk'

    @staticmethod
    def product_name(disk_product: str, redundancy_type: str, location: str) -> str:
        return azure_disk_product(disk_product, redundancy_type, location)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AzureStaticSizedDiskResource':
        assert data['type'] == AzureStaticSizedDiskResource.TYPE
        return AzureStaticSizedDiskResource(data['name'], data['storage_in_gib'])

    @staticmethod
    def create(
        product_versions: ProductVersions,
        disk_type: str,
        storage_in_gib: int,
        location: str,
        redundancy_type: str = 'LRS',
    ) -> 'AzureStaticSizedDiskResource':
        assert redundancy_type in ('LRS', 'ZRS'), redundancy_type

        # Azure bills for specific disk sizes so we must round the storage_in_gib to the nearest power of two
        disk = azure_disk_from_storage_in_gib(disk_type, storage_in_gib)
        assert disk, f'disk_type={disk_type} storage_in_gib={storage_in_gib}'

        product = AzureStaticSizedDiskResource.product_name(disk.name, redundancy_type, location)
        name = product_versions.resource_name(product)
        assert name, product
        return AzureStaticSizedDiskResource(name, disk.size_in_gib)

    def __init__(self, name: str, storage_in_gib: int):
        self.name = name
        self.storage_in_gib = storage_in_gib

    def to_dict(self) -> dict:
        return {
            'type': self.TYPE,
            'name': self.name,
            'storage_in_gib': self.storage_in_gib,
            'format_version': self.FORMAT_VERSION,
        }


class AzureDynamicSizedDiskResource(DynamicSizedDiskResourceMixin, AzureResource):
    FORMAT_VERSION = 1
    TYPE = 'azure_dynamic_sized_disk'

    @staticmethod
    def product_name(disk_product: str, redundancy_type: str, location: str) -> str:
        return azure_disk_product(disk_product, redundancy_type, location)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AzureDynamicSizedDiskResource':
        assert data['type'] == AzureDynamicSizedDiskResource.TYPE
        return AzureDynamicSizedDiskResource(data['disk_type'], data['location'], data['latest_disk_versions'])

    @staticmethod
    def create(
        product_versions: ProductVersions,
        disk_type: str,
        location: str,
        redundancy_type: str = 'LRS',
    ) -> 'AzureDynamicSizedDiskResource':
        assert redundancy_type in ('LRS', 'ZRS'), redundancy_type

        disk_name_to_resource_names = {}
        for disk in azure_disks_by_disk_type[disk_type]:
            disk_name = disk.name
            product = AzureDynamicSizedDiskResource.product_name(disk_name, redundancy_type, location)
            resource_name = product_versions.resource_name(product)
            assert resource_name, product
            disk_name_to_resource_names[disk_name] = resource_name
        return AzureDynamicSizedDiskResource(disk_type, location, disk_name_to_resource_names)

    def __init__(self, disk_type: str, location: str, disk_name_to_resource_names: Dict[str, str]):
        self.disk_type = disk_type
        self.location = location
        self.disk_name_to_resource_names = disk_name_to_resource_names

    def to_quantified_resource(
        self, cpu_in_mcpu: int, memory_in_bytes: int, worker_fraction_in_1024ths: int, external_storage_in_gib: int
    ) -> Optional[QuantifiedResource]:  # pylint: disable=unused-argument
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
            'format_version': self.FORMAT_VERSION,
        }


class AzureVMResource(VMResourceMixin, AzureResource):
    FORMAT_VERSION = 1
    TYPE = 'azure_vm'

    @staticmethod
    def product_name(machine_type: str, preemptible: bool, location: str) -> str:
        preemptible_str = 'spot' if preemptible else 'regular'
        return f'az/vm/{machine_type}/{preemptible_str}/{location}'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AzureVMResource':
        assert data['type'] == AzureVMResource.TYPE
        return AzureVMResource(data['name'])

    @staticmethod
    def create(
        product_versions: ProductVersions,
        machine_type: str,
        preemptible: bool,
        location: str,
    ) -> 'AzureVMResource':
        product = AzureVMResource.product_name(machine_type, preemptible, location)
        name = product_versions.resource_name(product)
        assert name, product
        return AzureVMResource(name)

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict:
        return {'type': self.TYPE, 'name': self.name, 'format_version': self.FORMAT_VERSION}


class AzureServiceFeeResource(ServiceFeeResourceMixin, AzureResource):
    FORMAT_VERSION = 1
    TYPE = 'azure_service_fee'

    @staticmethod
    def product_name():
        return 'az/service-fee'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AzureServiceFeeResource':
        assert data['type'] == AzureServiceFeeResource.TYPE
        return AzureServiceFeeResource(data['name'])

    @staticmethod
    def create(product_versions: ProductVersions) -> 'AzureServiceFeeResource':
        product = AzureServiceFeeResource.product_name()
        name = product_versions.resource_name(product)
        assert name, product
        return AzureServiceFeeResource(name)

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict:
        return {'type': self.TYPE, 'name': self.name, 'format_version': self.FORMAT_VERSION}


class AzureIPFeeResource(IPFeeResourceMixin, AzureResource):
    FORMAT_VERSION = 1
    TYPE = 'azure_ip_fee'

    @staticmethod
    def product_name(base: int):
        return f'az/ip-fee/{base}'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AzureIPFeeResource':
        assert data['type'] == AzureIPFeeResource.TYPE
        return AzureIPFeeResource(data['name'])

    @staticmethod
    def create(product_versions: ProductVersions, base: int) -> 'AzureIPFeeResource':
        product = AzureIPFeeResource.product_name(base)
        name = product_versions.resource_name(product)
        assert name, product
        return AzureIPFeeResource(name)

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict:
        return {'type': self.TYPE, 'name': self.name, 'format_version': self.FORMAT_VERSION}


def azure_resource_from_dict(data: dict) -> AzureResource:
    typ = data['type']
    if typ == AzureStaticSizedDiskResource.TYPE:
        return AzureStaticSizedDiskResource.from_dict(data)
    if typ == AzureDynamicSizedDiskResource.TYPE:
        return AzureDynamicSizedDiskResource.from_dict(data)
    if typ == AzureVMResource.TYPE:
        return AzureVMResource.from_dict(data)
    if typ == AzureServiceFeeResource.TYPE:
        return AzureServiceFeeResource.from_dict(data)
    assert typ == AzureIPFeeResource.TYPE
    return AzureIPFeeResource.from_dict(data)
