from abc import ABC, abstractmethod
from functools import wraps
import os
import glob
import google.oauth2.service_account

from hailtop.utils import blocking_to_async

from .google_storage import GCS


wildcards = ('*', '?', '[', ']', '{', '}')


# need a custom escape because escaped characters are not treated properly with glob.escape
def escape(path):
    new_path = []
    n = len(path)
    i = 0
    while i < n:
        if i < n - 1 and path[i] == '\\' and path[i + 1] in wildcards:
            new_path.append('[')
            new_path.append(path[i + 1])
            new_path.append(']')
            i += 2
            continue

        new_path.append(path[i])
        i += 1
    return ''.join(new_path)


class FileSystem(ABC):
    def __init__(self, thread_pool=None):
        self.thread_pool = thread_pool
        self._wrapped_listdir = self._wrap(self._listdir)
        self._wrapped_mkdir = self._wrap(self._mkdir)
        self._wrapped_glob = self._wrap(self._glob)
        self._wrapped_touch = self._wrap(self._touch)
        self._wrapped_exists = self._wrap(self._exists)
        self._wrapped_isfile = self._wrap(self._isfile)
        self._wrapped_isdir = self._wrap(self._isdir)

    def _wrap(self, fun):
        @wraps(fun)
        async def wrapped(*args, **kwargs):
            return await blocking_to_async(self.thread_pool,
                                           fun,
                                           *args,
                                           **kwargs)
        return wrapped

    async def listdir(self, path):
        return await self._wrapped_listdir(path)

    async def mkdir(self, path, create_parents=False):
        return await self._wrapped_mkdir(path, create_parents=create_parents)

    async def glob(self, pattern, recursive=False):
        return await self._wrapped_glob(pattern, recursive=recursive)

    async def touch(self, path, size=0):
        return await self._wrapped_touch(path, size=size)

    async def exists(self, path):
        return await self._wrapped_exists(path)

    async def isfile(self, path):
        return await self._wrapped_isfile(path)

    async def isdir(self, path):
        return await self._wrapped_isdir(path)

    @abstractmethod
    def _listdir(self, path):
        pass

    @abstractmethod
    def _mkdir(self, path, create_parents=False):
        pass

    @abstractmethod
    def _glob(self, pattern, recursive=False):
        pass

    @abstractmethod
    def _touch(self, path, size=0):
        pass

    @abstractmethod
    def _exists(self, path):
        pass

    @abstractmethod
    def _isfile(self, path):
        pass

    @abstractmethod
    def _isdir(self, path):
        pass


class LocalFileSystem(FileSystem):
    def __init__(self, thread_pool):
        super().__init__(thread_pool)

    def _listdir(self, path):
        return os.listdir(path)

    def _mkdir(self, path, create_parents=False):
        if create_parents:
            os.makedirs(path, exist_ok=True)
        else:
            os.mkdir(path)

    def _glob(self, pattern, recursive=False):
        # literal wildcards must be escaped in brackets \? => [?]
        pattern = escape(pattern)
        # glob.glob expands *, ? [], but ignores {}
        return glob.glob(pattern, recursive=recursive)

    def _touch(self, path, size=0):

        with open(path, 'ab') as fp:
            fp.truncate(size)

    def _exists(self, path):
        return os.path.exists(path)

    def _isfile(self, path):
        return os.path.isfile(path)

    def _isdir(self, path):
        return os.path.isdir(path)


class GoogleFileSystem(FileSystem):
    def __init__(self, thread_pool, key_file, project):
        super().__init__(thread_pool)
        credentials = google.oauth2.service_account.Credentials.from_service_account_file(key_file)
        self.gcs = GCS(thread_pool, project=project, credentials=credentials)

    def _listdir(self, path):
        raise NotImplementedError

    def _mkdir(self, path, create_parents=False):
        raise NotImplementedError

    def _glob(self, pattern, recursive=False):
        raise NotImplementedError

    def _touch(self, path, size=0):
        raise NotImplementedError

    def _exists(self, path):
        raise NotImplementedError

    def _isfile(self, path):
        raise NotImplementedError

    def _isdir(self, path):
        raise NotImplementedError
