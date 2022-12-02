from typing import Optional
import asyncio
import logging
import weakref


log = logging.getLogger('aiotools.tasks')


class TaskManagerClosedError(Exception):
    pass


class BackgroundTaskManager:
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        if loop is None:
            loop = asyncio.get_event_loop()

        self.tasks: weakref.WeakSet = weakref.WeakSet()
        self.loop = loop

        self._closed = False

    def ensure_future(self, coroutine) -> asyncio.Future:
        if self._closed:
            raise TaskManagerClosedError
        t = asyncio.ensure_future(coroutine)
        self.tasks.add(t)
        return t

    def ensure_future_threadsafe(self, coroutine):
        if self._closed:
            raise TaskManagerClosedError
        self.tasks.add(asyncio.run_coroutine_threadsafe(coroutine, self.loop))

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
