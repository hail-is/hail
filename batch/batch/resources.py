import abc
from typing import Any, Dict, Optional, TypedDict


class QuantifiedResource(TypedDict):
    name: str
    quantity: int


class Resource(abc.ABC):
    name: str

    @property
    def product(self):
        return self.name.rsplit('/', maxsplit=1)[0]

    @property
    def version(self):
        return self.name.rsplit('/', maxsplit=1)[1]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Resource':
        raise NotImplementedError

    @abc.abstractmethod
    def to_quantified_resource(
        self, cpu_in_mcpu: int, memory_in_bytes: int, worker_fraction_in_1024ths: int, external_storage_in_gib: int
    ) -> Optional[QuantifiedResource]:
        raise NotImplementedError

    @abc.abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError


class StaticSizedDiskResourceMixin(Resource, abc.ABC):
    storage_in_gib: int

    def to_quantified_resource(
        self, cpu_in_mcpu: int, memory_in_bytes: int, worker_fraction_in_1024ths: int, external_storage_in_gib: int
    ) -> Optional[QuantifiedResource]:  # pylint: disable=unused-argument
        del cpu_in_mcpu, memory_in_bytes, external_storage_in_gib
        # the factors of 1024 cancel between GiB -> MiB and fraction_1024 -> fraction
        return {'name': self.name, 'quantity': self.storage_in_gib * worker_fraction_in_1024ths}


class DynamicSizedDiskResourceMixin(Resource, abc.ABC):
    @abc.abstractmethod
    def to_quantified_resource(
        self, cpu_in_mcpu: int, memory_in_bytes: int, worker_fraction_in_1024ths: int, external_storage_in_gib: int
    ) -> Optional[QuantifiedResource]:  # pylint: disable=unused-argument
        raise NotImplementedError


class ComputeResourceMixin(Resource, abc.ABC):
    def to_quantified_resource(
        self, cpu_in_mcpu: int, memory_in_bytes: int, worker_fraction_in_1024ths: int, external_storage_in_gib: int
    ) -> Optional[QuantifiedResource]:  # pylint: disable=unused-argument
        del memory_in_bytes, worker_fraction_in_1024ths, external_storage_in_gib
        return {'name': self.name, 'quantity': cpu_in_mcpu}


class VMResourceMixin(Resource, abc.ABC):
    def to_quantified_resource(
        self, cpu_in_mcpu: int, memory_in_bytes: int, worker_fraction_in_1024ths: int, external_storage_in_gib: int
    ) -> Optional[QuantifiedResource]:  # pylint: disable=unused-argument
        del cpu_in_mcpu, memory_in_bytes, external_storage_in_gib
        return {'name': self.name, 'quantity': worker_fraction_in_1024ths}


class MemoryResourceMixin(Resource, abc.ABC):
    def to_quantified_resource(
        self, cpu_in_mcpu: int, memory_in_bytes: int, worker_fraction_in_1024ths: int, external_storage_in_gib: int
    ) -> Optional[QuantifiedResource]:  # pylint: disable=unused-argument
        del cpu_in_mcpu, worker_fraction_in_1024ths, external_storage_in_gib
        return {'name': self.name, 'quantity': memory_in_bytes // 1024 // 1024}


class IPFeeResourceMixin(Resource, abc.ABC):
    def to_quantified_resource(
        self, cpu_in_mcpu: int, memory_in_bytes: int, worker_fraction_in_1024ths: int, external_storage_in_gib: int
    ) -> Optional[QuantifiedResource]:  # pylint: disable=unused-argument
        del cpu_in_mcpu, memory_in_bytes, external_storage_in_gib
        return {'name': self.name, 'quantity': worker_fraction_in_1024ths}


class ServiceFeeResourceMixin(Resource, abc.ABC):
    def to_quantified_resource(
        self, cpu_in_mcpu: int, memory_in_bytes: int, worker_fraction_in_1024ths: int, external_storage_in_gib: int
    ) -> Optional[QuantifiedResource]:  # pylint: disable=unused-argument
        del memory_in_bytes, worker_fraction_in_1024ths, external_storage_in_gib
        return {'name': self.name, 'quantity': cpu_in_mcpu}
