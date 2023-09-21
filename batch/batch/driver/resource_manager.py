import abc
import logging
from typing import Any, List, Tuple

from hailtop.utils import time_msecs

from ..file_store import FileStore
from ..instance_config import InstanceConfig, QuantifiedResource
from .instance import Instance

log = logging.getLogger('compute_manager')


class VMDoesNotExist(Exception):
    pass


class VMState:
    pass


class NoTimestampVMState(VMState):
    def __init__(self, state: str, full_spec: Any):
        self.state = state
        self.full_spec = full_spec

    def __str__(self):
        return f'state={self.state} full_spec={self.full_spec}'


class UnknownVMState(NoTimestampVMState):
    def __init__(self, full_spec: Any):
        super().__init__('Unknown', full_spec)


class VMStateTerminated(NoTimestampVMState):
    def __init__(self, full_spec: Any):
        super().__init__('Terminated', full_spec)


class TimestampedVMState(VMState):
    def __init__(self, state: str, full_spec: Any, last_state_change_timestamp_msecs: int):
        assert last_state_change_timestamp_msecs is not None
        self.state = state
        self.full_spec = full_spec
        self.last_state_change_timestamp_msecs = last_state_change_timestamp_msecs

    def time_since_last_state_change(self) -> int:
        return time_msecs() - self.last_state_change_timestamp_msecs

    def __str__(self):
        return f'state={self.state} full_spec={self.full_spec} last_state_change_timestamp_msecs={self.last_state_change_timestamp_msecs}'


class VMStateCreating(TimestampedVMState):
    def __init__(self, full_spec: Any, last_state_change_timestamp_msecs: int):
        super().__init__('Creating', full_spec, last_state_change_timestamp_msecs)


class VMStateRunning(TimestampedVMState):
    def __init__(self, full_spec: Any, last_state_change_timestamp_msecs: int):
        super().__init__('Running', full_spec, last_state_change_timestamp_msecs)


class CloudResourceManager:
    @abc.abstractmethod
    def machine_type(self, cores: int, worker_type: str, local_ssd: bool) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def worker_type_and_cores(self, machine_type: str) -> Tuple[str, int]:
        raise NotImplementedError

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @abc.abstractmethod
    async def delete_vm(self, instance: Instance):
        raise NotImplementedError

    @abc.abstractmethod
    async def get_vm_state(self, instance: Instance) -> VMState:
        raise NotImplementedError
