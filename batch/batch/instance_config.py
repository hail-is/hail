import abc
from typing import Any, Dict

from .cloud.resource_utils import cores_mcpu_to_memory_bytes


def is_power_two(n):
    return (n & (n - 1) == 0) and n != 0


class InstanceConfig(abc.ABC):
    cloud: str
    version: int
    cores: int
    preemptible: bool
    local_ssd_data_disk: bool
    data_disk_size_gb: int
    job_private: bool
    worker_type: str
    machine_type: str
    location: str
    config: Dict[str, Any]

    @abc.abstractmethod
    def resources(self, cpu_in_mcpu, memory_in_bytes, storage_in_gib):
        pass

    def is_valid_configuration(self, valid_resources):
        is_valid = True
        dummy_resources = self.resources(0, 0, 0)
        for resource in dummy_resources:
            is_valid &= resource['name'] in valid_resources
        return is_valid

    def _cost_per_hour_from_resources(self, resource_rates, resources):  # pylint: disable=no-self-use
        cost_per_msec = 0
        for r in resources:
            name = r['name']
            quantity = r['quantity']
            rate_unit_msec = resource_rates[name]
            cost_per_msec += quantity * rate_unit_msec
        return cost_per_msec * 1000 * 60 * 60

    def cost_per_hour(self, resource_rates, cpu_in_mcpu, memory_in_bytes, storage_in_gb):
        resources = self.resources(cpu_in_mcpu, memory_in_bytes, storage_in_gb)
        return self._cost_per_hour_from_resources(resource_rates, resources)

    def cost_per_hour_from_cores(self, resource_rates, utilized_cores_mcpu):
        assert 0 <= utilized_cores_mcpu <= self.cores * 1000
        memory_in_bytes = cores_mcpu_to_memory_bytes(self.cloud, utilized_cores_mcpu, self.worker_type)
        storage_in_gb = 0   # we don't need to account for external storage
        return self.cost_per_hour(resource_rates, utilized_cores_mcpu, memory_in_bytes, storage_in_gb)

    def actual_cost_per_hour(self, resource_rates):
        cpu_in_mcpu = self.cores * 1000
        memory_in_bytes = cores_mcpu_to_memory_bytes(self.cloud, cpu_in_mcpu, self.worker_type)
        storage_in_gb = 0   # we don't need to account for external storage
        resources = self.resources(cpu_in_mcpu, memory_in_bytes, storage_in_gb)
        resources = [r for r in resources if 'service-fee' not in r['name']]
        return self._cost_per_hour_from_resources(resource_rates, resources)
