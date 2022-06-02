from typing import Optional
import asyncio
import logging
import weakref


log = logging.getLogger('aiotools.tasks')


class BackgroundTaskManager:
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        if loop is None:
            loop = asyncio.get_event_loop()

        self.tasks: weakref.WeakSet = weakref.WeakSet()
        self.loop = loop

    def ensure_future(self, coroutine) -> asyncio.Future:
        t = asyncio.ensure_future(coroutine)
        self.tasks.add(t)
        return t

    def ensure_future_threadsafe(self, coroutine):
        self.tasks.add(asyncio.run_coroutine_threadsafe(coroutine, self.loop))

    def shutdown(self):
        for task in self.tasks:
            try:
                task.cancel()
            except Exception:
                log.warning(f'encountered an exception while cancelling background task: {task}', exc_info=True)
