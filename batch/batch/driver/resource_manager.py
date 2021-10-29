from typing import Dict, List, Any, Tuple
import abc
import logging

from hailtop.utils import time_msecs

from .instance import Instance


log = logging.getLogger('compute_manager')


class VMDoesNotExist(Exception):
    pass


class VMState:
    pass


class UnknownVMState(VMState):
    def __init__(self, full_spec: Any):
        self.full_spec = full_spec

    def __str__(self):
        return f'state=Unknown full_spec={self.full_spec}'


class KnownVMState(VMState):
    def __init__(self, state: str, full_spec: Any, last_state_change_timestamp_msecs: int):
        assert last_state_change_timestamp_msecs is not None
        self.state = state
        self.full_spec = full_spec
        self.last_state_change_timestamp_msecs = last_state_change_timestamp_msecs

    def time_since_last_state_change(self) -> int:
        return time_msecs() - self.last_state_change_timestamp_msecs

    def __str__(self):
        return f'state={self.state} full_spec={self.full_spec} last_state_change_timestamp_msecs={self.last_state_change_timestamp_msecs}'


class VMStateCreating(KnownVMState):
    def __init__(self, full_spec: Any, last_state_change_timestamp_msecs: int):
        super().__init__('Creating', full_spec, last_state_change_timestamp_msecs)


class VMStateRunning(KnownVMState):
    def __init__(self, full_spec: Any, last_state_change_timestamp_msecs: int):
        super().__init__('Running', full_spec, last_state_change_timestamp_msecs)


class VMStateTerminated(KnownVMState):
    def __init__(self, full_spec: Any, last_state_change_timestamp_msecs: int):
        super().__init__('Terminated', full_spec, last_state_change_timestamp_msecs)


class CloudResourceManager:
    @abc.abstractmethod
    def machine_type(self, cores: int, worker_type: str):
        raise NotImplementedError

    @abc.abstractmethod
    def worker_type_cores(self, machine_type: str) -> Tuple[str, int]:
        raise NotImplementedError

    @abc.abstractmethod
    async def create_vm(self,
                        app,
                        machine_name: str,
                        activation_token: str,
                        max_idle_time_msecs: int,
                        worker_local_ssd_data_disk: bool,
                        worker_pd_ssd_data_disk_size_gb: int,
                        boot_disk_size_gb: int,
                        preemptible: bool,
                        job_private: bool,
                        location: str,
                        machine_type: str,
                        ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    async def delete_vm(self, instance: Instance):
        raise NotImplementedError

    @abc.abstractmethod
    async def get_vm_state(self, instance: Instance) -> VMState:
        raise NotImplementedError
