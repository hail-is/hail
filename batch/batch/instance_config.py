import abc
from typing import Dict, List, Sequence

from .driver.billing_manager import ProductVersions
from .resources import QuantifiedResource, Resource


def is_power_two(n):
    return (n & (n - 1) == 0) and n != 0


class InstanceConfig(abc.ABC):
    cloud: str
    cores: int
    job_private: bool
    resources: Sequence[Resource]

    @staticmethod
    @abc.abstractmethod
    def create(
        product_versions: ProductVersions,
        machine_type: str,
        preemptible: bool,
        local_ssd_data_disk: bool,
        data_disk_size_gb: int,
        boot_disk_size_gb: int,
        job_private: bool,
        location: str,
    ) -> 'InstanceConfig':
        raise NotImplementedError

    @abc.abstractmethod
    def worker_type(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError

    # FIXME: This should just be a `region` property on the instance_config
    # instead of accepting a "location"
    @abc.abstractmethod
    def region_for(self, location: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def instance_memory(self) -> int:
        raise NotImplementedError

    def quantified_resources(
        self,
        cpu_in_mcpu: int,
        memory_in_bytes: int,
        extra_storage_in_gib: int,
    ) -> List[QuantifiedResource]:
        assert memory_in_bytes % (1024 * 1024) == 0, memory_in_bytes
        assert isinstance(extra_storage_in_gib, int), extra_storage_in_gib
        if not self.job_private:
            assert is_power_two(self.cores) and self.cores <= 256, self.cores

        # FIXME: Only valid up to cores = 64
        worker_fraction_in_1024ths = 1024 * cpu_in_mcpu // (self.cores * 1000)

        _quantified_resources = []
        for resource in self.resources:
            quantified_resource = resource.to_quantified_resource(
                cpu_in_mcpu=cpu_in_mcpu,
                memory_in_bytes=memory_in_bytes,
                worker_fraction_in_1024ths=worker_fraction_in_1024ths,
                external_storage_in_gib=extra_storage_in_gib,
            )
            if quantified_resource is not None:
                _quantified_resources.append(quantified_resource)
        return _quantified_resources

    def is_valid_configuration(self, valid_resources):
        is_valid = True
        dummy_resources = self.quantified_resources(0, 0, 0)
        for resource in dummy_resources:
            is_valid &= resource['name'] in valid_resources
        return is_valid

    @staticmethod
    def _cost_per_hour_from_resources(resource_rates: Dict[str, float], resources: List[QuantifiedResource]) -> float:
        cost_per_msec = 0.0
        for r in resources:
            name = r['name']
            quantity = r['quantity']
            rate_unit_msec = resource_rates[name]
            cost_per_msec += quantity * rate_unit_msec
        return cost_per_msec * 1000 * 60 * 60

    # The price a user pays to rent a portion of this instance
    def price_per_hour(
        self,
        resource_rates: Dict[str, float],
        cpu_in_mcpu: int,
        memory_in_bytes: int,
        storage_in_gb: int,
    ) -> float:
        resources_billed_to_user = self.quantified_resources(cpu_in_mcpu, memory_in_bytes, storage_in_gb)
        return InstanceConfig._cost_per_hour_from_resources(resource_rates, resources_billed_to_user)

    def entire_instance_price_per_hour(
        self,
        resource_rates: Dict[str, float],
    ) -> float:
        return self.price_per_hour(
            resource_rates,
            self.cores * 1000,
            self.instance_memory(),
            0,  # we don't need to account for external storage
        )

    # The cost of procuring this instance from the cloud provider
    def cost_per_hour(self, resource_rates: Dict[str, float]) -> float:
        cpu_in_mcpu = self.cores * 1000
        memory_in_bytes = self.instance_memory()
        storage_in_gb = 0  # we don't need to account for external storage
        resources_billed_to_user = self.quantified_resources(cpu_in_mcpu, memory_in_bytes, storage_in_gb)
        resources_charged_by_the_cloud = [r for r in resources_billed_to_user if 'service-fee' not in r['name']]
        return InstanceConfig._cost_per_hour_from_resources(resource_rates, resources_charged_by_the_cloud)
