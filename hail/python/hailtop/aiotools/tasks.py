from typing import Iterable, Set
import asyncio
import logging
from contextlib import ExitStack


log = logging.getLogger('aiotools.tasks')


async def cancel_and_retrieve_all_exceptions(tasks: Iterable[asyncio.Task]):
    for task in tasks:
        if not task.cancelled():
            task.cancel()
    await asyncio.wait(tasks)
    with ExitStack() as retrieve_all_exceptions:
        # NB: only the first exception is raised
        for task in tasks:
            assert task.done()
            if not task.cancelled():
                retrieve_all_exceptions.callback(task.result)


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
