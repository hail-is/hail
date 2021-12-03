import abc
from typing import Dict, Any, Optional

from ...driver.billing_manager import ProductVersions
from ...resources import (Resource, DiskResourceMixin, ComputeResourceMixin, MemoryResourceMixin, IPFeeResourceMixin, ServiceFeeResourceMixin,
                          ExternalDiskResourceMixin, QuantifiedResource)


class GCPResource(Resource, abc.ABC):
    pass


def gcp_disk_product(disk_type: str) -> str:
    return f'disk/{disk_type}'


class GCPDiskResource(DiskResourceMixin, GCPResource):
    FORMAT_VERSION = 1
    TYPE = 'gcp_disk'

    @staticmethod
    def product_name(disk_type: str) -> str:
        return gcp_disk_product(disk_type)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'GCPDiskResource':
        assert data['type'] == GCPDiskResource.TYPE
        return GCPDiskResource(data['name'], data['storage_in_gib'])

    @staticmethod
    def new_resource(product_versions: ProductVersions,
                     disk_type: str,
                     storage_in_gib: int,
                     ) -> 'GCPDiskResource':
        product = GCPDiskResource.product_name(disk_type)
        name = product_versions.resource_name(product)
        assert name, product
        return GCPDiskResource(name, storage_in_gib)

    def __init__(self, name: str, storage_in_gib: int):
        self.name = name
        self.storage_in_gib = storage_in_gib

    def to_dict(self) -> dict:
        return {
            'type': self.TYPE,
            'name': self.name,
            'storage_in_gib': self.storage_in_gib,
            'version': self.FORMAT_VERSION
        }


class GCPExternalDiskResource(ExternalDiskResourceMixin, GCPResource):
    FORMAT_VERSION = 1
    TYPE = 'gcp_external_disk'

    @staticmethod
    def product_name(disk_type: str) -> str:
        return gcp_disk_product(disk_type)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'GCPExternalDiskResource':
        assert data['type'] == GCPExternalDiskResource.TYPE
        return GCPExternalDiskResource(data['name'])

    @staticmethod
    def new_resource(product_versions: ProductVersions, disk_type: str) -> 'GCPExternalDiskResource':
        product = GCPExternalDiskResource.product_name(disk_type)
        name = product_versions.resource_name(product)
        assert name, product
        return GCPExternalDiskResource(name)

    def __init__(self, name: str):
        self.name = name

    def to_quantified_resource(self,
                               cpu_in_mcpu: int,
                               memory_in_bytes: int,
                               worker_fraction_in_1024ths: int,
                               external_storage_in_gib: int) -> Optional[QuantifiedResource]:  # pylint: disable=unused-argument
        del cpu_in_mcpu, memory_in_bytes, worker_fraction_in_1024ths
        if external_storage_in_gib == 0:
            return None
        return {'name': self.name, 'quantity': external_storage_in_gib * 1024}  # storage is in units of MiB

    def to_dict(self) -> dict:
        return {
            'type': self.TYPE,
            'name': self.name,
            'version': self.FORMAT_VERSION
        }


class GCPComputeResource(ComputeResourceMixin, GCPResource):
    FORMAT_VERSION = 1
    TYPE = 'gcp_compute'

    @staticmethod
    def product_name(instance_family: str, preemptible: bool) -> str:
        preemptible_str = 'preemptible' if preemptible else 'nonpreemptible'
        return f'compute/{instance_family}-{preemptible_str}'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'GCPComputeResource':
        assert data['type'] == GCPComputeResource.TYPE
        return GCPComputeResource(data['name'])

    @staticmethod
    def new_resource(product_versions: ProductVersions,
                     instance_family: str,
                     preemptible: bool,
                     ) -> 'GCPComputeResource':
        product = GCPComputeResource.product_name(instance_family, preemptible)
        name = product_versions.resource_name(product)
        assert name, product
        return GCPComputeResource(name)

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict:
        return {
            'type': self.TYPE,
            'name': self.name,
            'format_version': self.FORMAT_VERSION
        }


class GCPMemoryResource(MemoryResourceMixin, GCPResource):
    FORMAT_VERSION = 1
    TYPE = 'gcp_memory'

    @staticmethod
    def product_name(instance_family: str, preemptible: bool) -> str:
        preemptible_str = 'preemptible' if preemptible else 'nonpreemptible'
        return f'memory/{instance_family}-{preemptible_str}'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'GCPMemoryResource':
        assert data['type'] == GCPMemoryResource.TYPE
        return GCPMemoryResource(data['name'])

    @staticmethod
    def new_resource(product_versions: ProductVersions,
                     instance_family: str,
                     preemptible: bool,
                     ) -> 'GCPMemoryResource':
        product = GCPMemoryResource.product_name(instance_family, preemptible)
        name = product_versions.resource_name(product)
        assert name, product
        return GCPMemoryResource(name)

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict:
        return {
            'type': self.TYPE,
            'name': self.name,
            'format_version': self.FORMAT_VERSION
        }


class GCPServiceFeeResource(ServiceFeeResourceMixin, GCPResource):
    FORMAT_VERSION = 1
    TYPE = 'gcp_service_fee'

    @staticmethod
    def product_name() -> str:
        return 'service-fee'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'GCPServiceFeeResource':
        assert data['type'] == GCPServiceFeeResource.TYPE
        return GCPServiceFeeResource(data['name'])

    @staticmethod
    def new_resource(product_versions: ProductVersions) -> 'GCPServiceFeeResource':
        product = GCPServiceFeeResource.product_name()
        name = product_versions.resource_name(product)
        assert name, product
        return GCPServiceFeeResource(name)

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict:
        return {
            'type': self.TYPE,
            'name': self.name,
            'format_version': self.FORMAT_VERSION
        }


class GCPIPFeeResource(IPFeeResourceMixin, GCPResource):
    FORMAT_VERSION = 1
    TYPE = 'gcp_ip_fee'

    @staticmethod
    def product_name(base: int) -> str:
        return f'ip-fee/{base}'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'GCPIPFeeResource':
        assert data['type'] == GCPIPFeeResource.TYPE
        return GCPIPFeeResource(data['name'])

    @staticmethod
    def new_resource(product_versions: ProductVersions, base: int) -> 'GCPIPFeeResource':
        product = GCPIPFeeResource.product_name(base)
        name = product_versions.resource_name(product)
        assert name, product
        return GCPIPFeeResource(name)

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict:
        return {
            'type': self.TYPE,
            'name': self.name,
            'format_version': self.FORMAT_VERSION
        }


def gcp_resource_from_dict(data: dict) -> GCPResource:
    typ = data['type']
    if typ == GCPDiskResource.TYPE:
        return GCPDiskResource.from_dict(data)
    if typ == GCPExternalDiskResource.TYPE:
        return GCPExternalDiskResource.from_dict(data)
    if typ == GCPComputeResource.TYPE:
        return GCPComputeResource.from_dict(data)
    if typ == GCPMemoryResource.TYPE:
        return GCPMemoryResource.from_dict(data)
    if typ == GCPServiceFeeResource.TYPE:
        return GCPServiceFeeResource.from_dict(data)
    assert typ == GCPIPFeeResource.TYPE
    return GCPIPFeeResource.from_dict(data)
