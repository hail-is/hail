from typing import Iterable, Set
import asyncio
import logging
from ..utils import cancel_and_retrieve_all_exceptions


log = logging.getLogger('aiotools.tasks')


class TaskManagerClosedError(Exception):
    pass


class BackgroundTaskManager:
    def __init__(self):
        # These must be strong references so that tasks do not get GC'd
        # The event loop only keeps weak references to tasks
        self.tasks: Set[asyncio.Task] = set()
        self._closed = False

    def ensure_future(self, coroutine):
        if self._closed:
            raise TaskManagerClosedError
        t = asyncio.create_task(coroutine)

        def retieve_exception_and_remove_from_task_list(task: asyncio.Task):
            assert task.done()
            if not task.cancelled():
                if exc := task.exception():
                    log.exception(f'child task {task} of manager {self} raised', exc)
            self.tasks.remove(task)

        t.add_done_callback(retieve_exception_and_remove_from_task_list)
        self.tasks.add(t)

    async def shutdown(self):
        await cancel_and_retrieve_all_exceptions(self.tasks)
