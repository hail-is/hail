from typing import Set
import asyncio
import logging


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
        t.add_done_callback(lambda _: self.tasks.remove(t))
        self.tasks.add(t)

    def shutdown(self):
        self._closed = True
        for task in self.tasks:
            try:
                task.cancel()
            except Exception:
                log.warning(f'encountered an exception while cancelling background task: {task}', exc_info=True)

    async def shutdown_and_wait(self):
        self.shutdown()
        await asyncio.wait(self.tasks, return_when=asyncio.ALL_COMPLETED)
