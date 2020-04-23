import os
import shutil
import glob
from functools import wraps

from .utils import blocking_to_async, flatten


def make_parent_dirs(path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)


def cp(src, dest):  # pylint: disable=invalid-name
    make_parent_dirs(dest)
    shutil.copy(src, dest)


def remove(path):
    if os.path.exists(path):
        assert os.path.isfile(path)
        os.remove(path)


def touch(path, size=0):
    with open(path, 'ab') as fp:
        fp.truncate(size)


class AsyncOS:
    def __init__(self, thread_pool):
        self.thread_pool = thread_pool
        self._wrapped_make_parent_dirs = self._wrap(make_parent_dirs)
        self._wrapped_cp = self._wrap(cp)
        self._wrapped_listdir = self._wrap(os.listdir)
        self._wrapped_remove = self._wrap(remove)
        self._wrapped_touch = self._wrap(touch)
        self._wrapped_glob = self._wrap(glob.glob)

    def _wrap(self, fun):
        @wraps(fun)
        async def wrapped(*args, **kwargs):
            return await blocking_to_async(self.thread_pool,
                                           fun,
                                           *args,
                                           **kwargs)
        return wrapped

    async def make_parent_dirs(self, path):
        return await self._wrapped_make_parent_dirs(path)

    async def cp(self, src, dest):  # pylint: disable=invalid-name
        return await self._wrapped_cp(src, dest)

    async def listdir(self, path):
        return await self._wrapped_listdir(path)

    async def remove(self, path):
        return await self._wrapped_remove(path)

    async def touch(self, path, size):
        return await self._wrapped_touch(path, size)

    async def glob(self, pattern, recursive=False):
        return await self._wrapped_glob(pattern, recursive=recursive)
