import os
import shutil

from .utils import blocking_to_async


class AsyncOS:
    def __init__(self, thread_pool):
        self.thread_pool = thread_pool
        self._wrapped_makedirs = self._wrap(AsyncOS._makedirs)
        self._wrapped_cp = self._wrap(AsyncOS._cp)
        self._wrapped_listdir = self._wrap(AsyncOS._listdir)
        self._wrapped_remove = self._wrap(AsyncOS._remove)
        self._wrapped_new_file = self._wrap(AsyncOS._new_file)

    def _wrap(self, fun):
        async def wrapped(*args, **kwargs):
            return await blocking_to_async(self.thread_pool,
                                           fun,
                                           *args,
                                           **kwargs)
        wrapped.__name__ = fun.__name__
        return wrapped

    async def makedirs(self, path):
        return await self._wrapped_makedirs(self, path)

    async def cp(self, src, dest):
        return await self._wrapped_cp(self, src, dest)

    async def listdir(self, path):
        return await self._wrapped_listdir(self, path)

    async def remove(self, path):
        return await self._wrapped_remove(self, path)

    async def new_file(self, path, size):
        return await self._wrapped_new_file(self, path, size)

    def _makedirs(self, path):
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _cp(self, src, dest):
        src = os.path.abspath(src)
        dest = os.path.abspath(dest)
        self._makedirs(dest)
        shutil.copy(src, dest)

    def _listdir(self, path):
        return os.listdir(path)

    def _remove(self, path):
        if os.path.exists(path):
            assert os.path.isfile(path)
            os.remove(path)

    def _new_file(self, path, size):
        with open(path, 'ab') as fp:
            fp.truncate(size)
