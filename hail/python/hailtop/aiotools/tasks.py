import asyncio
import logging
import weakref


log = logging.getLogger('aiotools.tasks')


class BackgroundTaskManager:
    def __init__(self):
        self.tasks: weakref.WeakSet = weakref.WeakSet()

    def ensure_future(self, coroutine):
        self.tasks.add(asyncio.ensure_future(coroutine))

    def shutdown(self):
        for task in self.tasks:
            try:
                task.cancel()
            except Exception:
                log.warning(f'encountered an exception while cancelling background task: {task}', exc_info=True)
