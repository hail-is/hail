from typing import Optional, Any, TYPE_CHECKING
import abc
import logging

from .instance import Instance

if TYPE_CHECKING:
    from .instance_collection import InstanceCollection  # pylint: disable=cyclic-import


log = logging.getLogger('compute_manager')


class VMDoesNotExist(Exception):
    pass


class VMState:
    UNKNOWN = 'Unknown'
    CREATING = 'Creating'
    RUNNING = 'Running'
    TERMINATED = 'Terminated'

    def __init__(self, state: str, full_spec: Any, last_state_change_timestamp_msecs: int = None):
        self.state = state
        self.full_spec = full_spec
        self.last_state_change_timestamp_msecs = last_state_change_timestamp_msecs

    def __str__(self):
        return f'state={self.state} full_spec={self.full_spec} last_state_change_timestamp_msecs={self.last_state_change_timestamp_msecs}'


class CloudResourceManager:
    default_location: str

    @abc.abstractmethod
    async def create_vm(self,
                        app,
                        inst_coll: 'InstanceCollection',
                        machine_name: str,
                        activation_token: str,
                        max_idle_time_msecs: int,
                        worker_local_ssd_data_disk: bool,
                        worker_pd_ssd_data_disk_size_gb: int,
                        boot_disk_size_gb: int,
                        preemptible: bool,
                        job_private: bool,
                        machine_type: Optional[str] = None,
                        worker_type: Optional[str] = None,
                        cores: Optional[int] = None,
                        location: Optional[str] = None,
                        ) -> Optional[Instance]:
        pass

    @abc.abstractmethod
    async def delete_vm(self, instance: Instance):
        pass

    @abc.abstractmethod
    async def get_vm_state(self, instance: Instance) -> VMState:
        pass
