import logging
from typing import List

from ....driver.instance import Instance
from ....driver.resource_manager import CloudResourceManager, VMState
from ....file_store import FileStore
from ....instance_config import InstanceConfig, QuantifiedResource

log = logging.getLogger('resource_manager')


# TODO: implement with an EC2 client.
class AWSResourceManager(CloudResourceManager):
    def machine_type(self, cores: int, worker_type: str, local_ssd: bool) -> str:
        raise NotImplementedError

    def instance_config(
        self,
        machine_type: str,
        preemptible: bool,
        local_ssd_data_disk: bool,
        data_disk_size_gb: int,
        boot_disk_size_gb: int,
        job_private: bool,
        location: str,
    ) -> InstanceConfig:
        raise NotImplementedError

    async def create_vm(
        self,
        file_store: FileStore,
        machine_name: str,
        activation_token: str,
        max_idle_time_msecs: int,
        local_ssd_data_disk: bool,
        data_disk_size_gb: int,
        boot_disk_size_gb: int,
        preemptible: bool,
        job_private: bool,
        location: str,
        machine_type: str,
        instance_config: InstanceConfig,
    ) -> List[QuantifiedResource]:
        raise NotImplementedError

    async def delete_vm(self, instance: Instance):
        raise NotImplementedError

    async def get_vm_state(self, instance: Instance) -> VMState:
        raise NotImplementedError
