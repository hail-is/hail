from typing import Callable, Set
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
        t.add_done_callback(self.on_task_done(t))
        self.tasks.add(t)

    def on_task_done(self, t: asyncio.Task) -> Callable[[asyncio.Future], None]:
        def callback(fut: asyncio.Future):
            self.tasks.remove(t)
            try:
                if e := fut.exception():
                    log.exception(e)
            except asyncio.CancelledError:
                if not self._closed:
                    log.exception('Background task was cancelled before task manager shutdown')

        return callback

    def shutdown(self):
        self._closed = True
        for task in self.tasks:
            try:
                task.cancel()
            except Exception:
                log.warning(f'encountered an exception while cancelling background task: {task}', exc_info=True)

    async def shutdown_and_wait(self):
        self.shutdown()
        if self.tasks:
            await asyncio.wait(self.tasks, return_when=asyncio.ALL_COMPLETED)
