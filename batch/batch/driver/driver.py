import abc
import datetime
from typing import Optional, Callable, Awaitable

from gear import Database

from .resource_manager import CloudResourceManager
from .instance_collection_manager import InstanceCollectionManager
from ..inst_coll_config import InstanceCollectionConfigs


async def process_outstanding_events(db: Database, process_events_since: Callable[[str], Awaitable[str]]):
    row = await db.select_and_fetchone('SELECT * FROM `events_mark`;')

    mark = row['mark']
    if mark is None:
        mark = datetime.datetime.utcnow().isoformat() + 'Z'
        await db.execute_update('UPDATE `events_mark` SET mark = %s;', (mark,))

    mark = await process_events_since(mark)

    if mark is not None:
        await db.execute_update('UPDATE `events_mark` SET mark = %s;', (mark,))


class CloudDriver(abc.ABC):
    resource_manager: CloudResourceManager
    inst_coll_manager: InstanceCollectionManager

    @staticmethod
    @abc.abstractmethod
    async def create(app, machine_name_prefix: str, inst_coll_configs: InstanceCollectionConfigs,
                     credentials_file: Optional[str] = None) -> 'CloudDriver':
        pass

    @abc.abstractmethod
    async def shutdown(self):
        pass
