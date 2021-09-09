import abc
import datetime
from typing import Any, TYPE_CHECKING
import logging

import hailtop.aiotools as aiotools
from hailtop.utils import periodically_call
from gear import Database

if TYPE_CHECKING:
    from .compute_manager import BaseComputeManager  # pylint: disable=cyclic-import

log = logging.getLogger('activity_monitor')


class BaseActivityMonitor(abc.ABC):
    activity_logs_client: Any

    def __init__(self, compute_manager: 'BaseComputeManager'):
        app = compute_manager.app

        self.compute_manager = compute_manager
        self.app = app
        self.db: Database = app['db']
        self.zone_monitor = compute_manager.zone_monitor
        self.task_manager = aiotools.BackgroundTaskManager()

    async def async_init(self):
        self.task_manager.ensure_future(self.event_loop())

    async def shutdown(self):
        self.task_manager.shutdown()

    async def handle_preempt_event(self, instance, timestamp):
        await instance.inst_coll.call_delete_instance(instance, 'preempted', timestamp=timestamp)

    async def handle_delete_done_event(self, instance, timestamp):
        await instance.inst_coll.remove_instance(instance, 'deleted', timestamp)

    async def handle_call_delete_event(self, instance, timestamp):
        await instance.mark_deleted('deleted', timestamp)

    def handle_insert_done_event(self, zone: str, operation_id: str, success: bool):
        self.zone_monitor.zone_success_rate.push(zone, operation_id, success)

    @abc.abstractmethod
    async def handle_event(self, event):
        pass

    @abc.abstractmethod
    async def list_events(self, mark):
        pass

    @abc.abstractmethod
    def event_timestamp_msecs(self, event):
        pass

    async def handle_events(self):
        row = await self.db.select_and_fetchone('SELECT * FROM `events_mark`;')
        mark = row['mark']
        if mark is None:
            mark = datetime.datetime.utcnow().isoformat() + 'Z'
            await self.db.execute_update('UPDATE `events_mark` SET mark = %s;', (mark,))

        log.info(f'querying logging client with mark {mark}')
        start_mark = mark
        mark = None
        async for event in await self.list_events(start_mark):
            # take the last, largest timestamp
            mark = self.event_timestamp_msecs(event)
            await self.handle_event(event)

        if mark is not None:
            await self.db.execute_update('UPDATE `events_mark` SET mark = %s;', (mark,))

    async def event_loop(self):
        await periodically_call(15, self.handle_events)
