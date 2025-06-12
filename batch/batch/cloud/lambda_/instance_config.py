from typing import List, Union

from ...driver.billing_manager import ProductVersions
from ...instance_config import InstanceConfig
from .resources import LambdaResource, lambda_resource_from_dict


LAMBDA_INSTANCE_CONFIG_VERSION = 1


class LambdaSlimInstanceConfig(InstanceConfig):
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
    ) -> 'LambdaSlimInstanceConfig':  # pylint: disable=unused-argument
        raise NotImplementedError

    def __init__(
        self,
        machine_type: str,
        preemptible: bool,
        local_ssd_data_disk: bool,
        data_disk_size_gb: int,
        boot_disk_size_gb: int,
        job_private: bool,
        resources: List[LambdaResource],
    ):
        self.cloud = 'lambda'
        self.machine_type = machine_type
        self.preemptible = preemptible
        self.local_ssd_data_disk = local_ssd_data_disk
        self.data_disk_size_gb = data_disk_size_gb
        self.boot_disk_size_gb = boot_disk_size_gb
        self.job_private = job_private
        self.resources = resources

    def worker_type(self) -> str:
        raise NotImplementedError

    def instance_memory(self) -> int:
        raise NotImplementedError

    def region_for(self, location: str) -> str:
        raise NotImplementedError

    @staticmethod
    def from_dict(data: dict) -> 'LambdaSlimInstanceConfig':
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {
            'version': LAMBDA_INSTANCE_CONFIG_VERSION,
            'cloud': 'lambda',
            'job_private': self.job_private,
            'resources': [resource.to_dict() for resource in self.resources],
        }
