import abc
import datetime
import logging
from typing import Awaitable, Callable, TYPE_CHECKING, Optional

from gear import Database

if TYPE_CHECKING:
    from ..instance_config import InstanceConfig
    from .instance import Instance  # pylint: disable=cyclic-import

log = logging.getLogger('compute_manager')


class VMDoesNotExist(Exception):
    pass


class VMState:
    CREATING = 'Creating'
    RUNNING = 'Running'
    TERMINATED = 'Terminated'

    def __init__(self, state, full_spec, last_start_timestamp=None):
        self.state = state
        self.full_spec = full_spec
        self.last_start_timestamp = last_start_timestamp


async def process_outstanding_events(db: Database, process_events_since: Callable[[str], Awaitable[str]]):
    row = await db.select_and_fetchone('SELECT * FROM `events_mark`;')

    mark = row['mark']
    if mark is None:
        mark = datetime.datetime.utcnow().isoformat() + 'Z'
        await db.execute_update('UPDATE `events_mark` SET mark = %s;', (mark,))

    mark = await process_events_since(mark)

    if mark is not None:
        await db.execute_update('UPDATE `events_mark` SET mark = %s;', (mark,))


class CloudResourceManager(abc.ABC):
    default_location: str
    cloud: str

    @abc.abstractmethod
    def prepare_vm(self,
                   app,
                   machine_name,
                   activation_token,
                   max_idle_time_msecs,
                   worker_local_ssd_data_disk,
                   worker_pd_ssd_data_disk_size_gb,
                   boot_disk_size_gb,
                   preemptible,
                   job_private,
                   machine_type: Optional[str] = None,
                   worker_type: Optional[str] = None,
                   cores: Optional[int] = None,
                   location: Optional[str] = None,
                   ) -> Optional['InstanceConfig']:
        pass

    @abc.abstractmethod
    async def create_vm(self, instance_config: 'InstanceConfig'):
        pass

    @abc.abstractmethod
    async def delete_vm(self, instance: 'Instance'):
        pass

    @abc.abstractmethod
    async def get_vm_state(self, instance: 'Instance') -> VMState:
        pass
