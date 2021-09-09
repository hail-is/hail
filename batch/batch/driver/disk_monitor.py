import abc
from typing import TYPE_CHECKING

import hailtop.aiotools as aiotools
from hailtop.utils import periodically_call

if TYPE_CHECKING:
    from .compute_manager import BaseComputeManager  # pylint: disable=cyclic-import


class BaseDiskMonitor(abc.ABC):
    def __init__(self, compute_manager: 'BaseComputeManager'):
        app = compute_manager.app
        self.app = app
        self.compute_manager = compute_manager
        self.task_manager = aiotools.BackgroundTaskManager()

    async def async_init(self):
        self.task_manager.ensure_future(self.delete_orphaned_disks_loop())

    async def shutdown(self):
        self.task_manager.shutdown()

    @abc.abstractmethod
    async def delete_orphaned_disks(self):
        pass

    async def delete_orphaned_disks_loop(self):
        await periodically_call(60, self.delete_orphaned_disks)
