import abc
import datetime
from typing import Awaitable, Callable

from gear import Database
from hailtop import aiotools

from ..inst_coll_config import InstanceCollectionConfigs
from .billing_manager import CloudBillingManager
from .instance_collection import InstanceCollectionManager


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
    inst_coll_manager: InstanceCollectionManager
    billing_manager: CloudBillingManager

    @staticmethod
    @abc.abstractmethod
    async def create(
        app,
        db: Database,
        machine_name_prefix: str,
        namespace: str,
        inst_coll_configs: InstanceCollectionConfigs,
        credentials_file: str,
        task_manager: aiotools.BackgroundTaskManager,
    ) -> 'CloudDriver':
        raise NotImplementedError

    @abc.abstractmethod
    async def shutdown(self):
        raise NotImplementedError
