from typing import Any, Dict, Optional

from ...driver.billing_manager import ProductVersions
from ...resources import (
    ComputeResourceMixin,
    DynamicSizedDiskResourceMixin,
    IPFeeResourceMixin,
    MemoryResourceMixin,
    QuantifiedResource,
    Resource,
    ServiceFeeResourceMixin,
    StaticSizedDiskResourceMixin,
    VMResourceMixin,
)


class GCPResource(Resource):
    pass


class GCPStaticSizedDiskResource(StaticSizedDiskResourceMixin, GCPResource):
    FORMAT_VERSION = 1
    TYPE = 'gcp_static_sized_disk'

    @staticmethod
    def product_name(disk_type: str, region: str) -> str:
        return f'disk/{disk_type}/{region}'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'GCPStaticSizedDiskResource':
        assert data['type'] == GCPStaticSizedDiskResource.TYPE
        return GCPStaticSizedDiskResource(data['name'], data['storage_in_gib'])

    @staticmethod
    def create(
        product_versions: ProductVersions, disk_type: str, storage_in_gib: int, region: str
    ) -> 'GCPStaticSizedDiskResource':
        product = GCPStaticSizedDiskResource.product_name(disk_type, region)
        if product_versions.latest_version(product) is None:
            product = f'disk/{disk_type}'  # backwards compatibility
        name = product_versions.resource_name(product)
        assert name, product
        return GCPStaticSizedDiskResource(name, storage_in_gib)

    def __init__(self, name: str, storage_in_gib: int):
        self.name = name
        self.storage_in_gib = storage_in_gib

    def to_dict(self) -> dict:
        return {
            'type': self.TYPE,
            'name': self.name,
            'storage_in_gib': self.storage_in_gib,
            'version': self.FORMAT_VERSION,
        }


class GCPDynamicSizedDiskResource(DynamicSizedDiskResourceMixin, GCPResource):
    FORMAT_VERSION = 1
    TYPE = 'gcp_dynamic_sized_disk'

    @staticmethod
    def product_name(disk_type: str, region: str) -> str:
        return f'disk/{disk_type}/{region}'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'GCPDynamicSizedDiskResource':
        assert data['type'] == GCPDynamicSizedDiskResource.TYPE
        return GCPDynamicSizedDiskResource(data['name'])

    @staticmethod
    def create(product_versions: ProductVersions, disk_type: str, region: str) -> 'GCPDynamicSizedDiskResource':
        product = GCPDynamicSizedDiskResource.product_name(disk_type, region)
        if product_versions.latest_version(product) is None:
            product = f'disk/{disk_type}'  # backwards compatibility
        name = product_versions.resource_name(product)
        assert name, product
        return GCPDynamicSizedDiskResource(name)

    def __init__(self, name: str):
        self.name = name

    def to_quantified_resource(
        self, cpu_in_mcpu: int, memory_in_bytes: int, worker_fraction_in_1024ths: int, external_storage_in_gib: int
    ) -> Optional[QuantifiedResource]:  # pylint: disable=unused-argument
        del cpu_in_mcpu, memory_in_bytes, worker_fraction_in_1024ths
        if external_storage_in_gib == 0:
            return None
        return {'name': self.name, 'quantity': external_storage_in_gib * 1024}  # storage is in units of MiB

    def to_dict(self) -> dict:
        return {'type': self.TYPE, 'name': self.name, 'version': self.FORMAT_VERSION}


class GCPLocalSSDStaticSizedDiskResource(StaticSizedDiskResourceMixin, GCPResource):
    FORMAT_VERSION = 1
    TYPE = 'gcp_local_ssd_static_sized_disk'

    @staticmethod
    def product_name(preemptible: bool, region: str) -> str:
        preemptible_str = 'preemptible' if preemptible else 'nonpreemptible'
        return f'disk/local-ssd/{preemptible_str}/{region}'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'GCPLocalSSDStaticSizedDiskResource':
        assert data['type'] == GCPLocalSSDStaticSizedDiskResource.TYPE
        return GCPLocalSSDStaticSizedDiskResource(data['name'], data['storage_in_gib'])

    @staticmethod
    def create(
        product_versions: ProductVersions,
        storage_in_gib: int,
        preemptible: bool,
        region: str,
    ) -> 'GCPLocalSSDStaticSizedDiskResource':
        product = GCPLocalSSDStaticSizedDiskResource.product_name(preemptible, region)
        if product_versions.latest_version(product) is None:
            product = 'disk/local-ssd'  # backwards compatibility
        name = product_versions.resource_name(product)
        assert name, product
        return GCPLocalSSDStaticSizedDiskResource(name, storage_in_gib)

    def __init__(self, name: str, storage_in_gib: int):
        self.name = name
        self.storage_in_gib = storage_in_gib

    def to_dict(self) -> dict:
        return {
            'type': self.TYPE,
            'name': self.name,
            'storage_in_gib': self.storage_in_gib,
            'version': self.FORMAT_VERSION,
        }


class GCPComputeResource(ComputeResourceMixin, GCPResource):
    FORMAT_VERSION = 1
    TYPE = 'gcp_compute'

    @staticmethod
    def product_name(instance_family: str, preemptible: bool, region: str) -> str:
        preemptible_str = 'preemptible' if preemptible else 'nonpreemptible'
        return f'compute/{instance_family}-{preemptible_str}/{region}'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'GCPComputeResource':
        assert data['type'] == GCPComputeResource.TYPE
        return GCPComputeResource(data['name'])

    @staticmethod
    def create(
        product_versions: ProductVersions,
        instance_family: str,
        preemptible: bool,
        region: str,
    ) -> 'GCPComputeResource':
        product = GCPComputeResource.product_name(instance_family, preemptible, region)
        if product_versions.latest_version(product) is None:
            preemptible_str = 'preemptible' if preemptible else 'nonpreemptible'
            product = f'compute/{instance_family}-{preemptible_str}'  # backwards compatibility
        name = product_versions.resource_name(product)
        assert name, product
        return GCPComputeResource(name)

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict:
        return {'type': self.TYPE, 'name': self.name, 'format_version': self.FORMAT_VERSION}


class GCPAcceleratorResource(VMResourceMixin, GCPResource):
    FORMAT_VERSION = 1
    TYPE = 'gcp_accelerator'

    @staticmethod
    def product_name(accelerator_family: str, preemptible: bool, region: str) -> str:
        preemptible_str = 'preemptible' if preemptible else 'nonpreemptible'
        return f'accelerator/{accelerator_family}-{preemptible_str}/{region}'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'GCPAcceleratorResource':
        assert data['type'] == GCPAcceleratorResource.TYPE
        return GCPAcceleratorResource(data['name'])

    @staticmethod
    def create(
        product_versions: ProductVersions,
        accelerator_family: str,
        preemptible: bool,
        region: str,
    ) -> 'GCPAcceleratorResource':
        product = GCPAcceleratorResource.product_name(accelerator_family, preemptible, region)
        name = product_versions.resource_name(product)
        assert name, product
        return GCPAcceleratorResource(name)

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict:
        return {'type': self.TYPE, 'name': self.name, 'format_version': self.FORMAT_VERSION}


class GCPMemoryResource(MemoryResourceMixin, GCPResource):
    FORMAT_VERSION = 1
    TYPE = 'gcp_memory'

    @staticmethod
    def product_name(instance_family: str, preemptible: bool, region: str) -> str:
        preemptible_str = 'preemptible' if preemptible else 'nonpreemptible'
        return f'memory/{instance_family}-{preemptible_str}/{region}'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'GCPMemoryResource':
        assert data['type'] == GCPMemoryResource.TYPE
        return GCPMemoryResource(data['name'])

    @staticmethod
    def create(
        product_versions: ProductVersions,
        instance_family: str,
        preemptible: bool,
        region: str,
    ) -> 'GCPMemoryResource':
        product = GCPMemoryResource.product_name(instance_family, preemptible, region)
        if product_versions.latest_version(product) is None:
            preemptible_str = 'preemptible' if preemptible else 'nonpreemptible'
            product = f'memory/{instance_family}-{preemptible_str}'  # backwards compatibility
        name = product_versions.resource_name(product)
        assert name, product
        return GCPMemoryResource(name)

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict:
        return {'type': self.TYPE, 'name': self.name, 'format_version': self.FORMAT_VERSION}


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
    def create(product_versions: ProductVersions) -> 'GCPServiceFeeResource':
        product = GCPServiceFeeResource.product_name()
        name = product_versions.resource_name(product)
        assert name, product
        return GCPServiceFeeResource(name)

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict:
        return {'type': self.TYPE, 'name': self.name, 'format_version': self.FORMAT_VERSION}


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
    def create(product_versions: ProductVersions, base: int) -> 'GCPIPFeeResource':
        product = GCPIPFeeResource.product_name(base)
        name = product_versions.resource_name(product)
        assert name, product
        return GCPIPFeeResource(name)

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict:
        return {'type': self.TYPE, 'name': self.name, 'format_version': self.FORMAT_VERSION}


class GCPSupportLogsSpecsAndFirewallFees(GCPResource):
    FORMAT_VERSION = 1
    TYPE = 'gcp_support_logs_specs_and_firewall_fees'

    def to_quantified_resource(
        self, cpu_in_mcpu: int, memory_in_bytes: int, worker_fraction_in_1024ths: int, external_storage_in_gib: int
    ) -> Optional[QuantifiedResource]:  # pylint: disable=unused-argument
        del memory_in_bytes, worker_fraction_in_1024ths, external_storage_in_gib
        return {'name': self.name, 'quantity': cpu_in_mcpu}

    @staticmethod
    def product_name() -> str:
        return 'gcp-support-logs-specs-and-firewall-fees'

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'GCPSupportLogsSpecsAndFirewallFees':
        assert data['type'] == GCPSupportLogsSpecsAndFirewallFees.TYPE
        return GCPSupportLogsSpecsAndFirewallFees(data['name'])

    @staticmethod
    def create(product_versions: ProductVersions) -> 'GCPSupportLogsSpecsAndFirewallFees':
        product = GCPSupportLogsSpecsAndFirewallFees.product_name()
        name = product_versions.resource_name(product)
        assert name, product
        return GCPSupportLogsSpecsAndFirewallFees(name)

    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> dict:
        return {'type': self.TYPE, 'name': self.name, 'format_version': self.FORMAT_VERSION}


def gcp_resource_from_dict(data: dict) -> GCPResource:
    typ = data['type']
    if typ == GCPStaticSizedDiskResource.TYPE:
        return GCPStaticSizedDiskResource.from_dict(data)
    if typ == GCPDynamicSizedDiskResource.TYPE:
        return GCPDynamicSizedDiskResource.from_dict(data)
    if typ == GCPLocalSSDStaticSizedDiskResource.TYPE:
        return GCPLocalSSDStaticSizedDiskResource.from_dict(data)
    if typ == GCPComputeResource.TYPE:
        return GCPComputeResource.from_dict(data)
    if typ == GCPMemoryResource.TYPE:
        return GCPMemoryResource.from_dict(data)
    if typ == GCPServiceFeeResource.TYPE:
        return GCPServiceFeeResource.from_dict(data)
    if typ == GCPAcceleratorResource.TYPE:
        return GCPAcceleratorResource.from_dict(data)
    if typ == GCPIPFeeResource.TYPE:
        return GCPIPFeeResource.from_dict(data)
    assert typ == GCPSupportLogsSpecsAndFirewallFees.TYPE, typ
    return GCPSupportLogsSpecsAndFirewallFees.from_dict(data)
