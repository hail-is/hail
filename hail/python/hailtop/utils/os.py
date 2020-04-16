import os
import shutil
import glob

from .utils import blocking_to_async, flatten


wildcards = ('*', '?', '[', ']', '{', '}')


# need a custom escape because escaped characters are not treated properly with glob.escape
# and fnmatch doesn't work with escaped characters like \?
def escape(path):
    new_path = []
    n = len(path)
    i = 0
    while i < n:
        if i <= n - 1 and path[i] == '\\' and path[i + 1] in wildcards:
            new_path.append('[')
            new_path.append(path[i + 1])
            new_path.append(']')
            i += 2
            continue

        new_path.append(path[i])
        i += 1
    return ''.join(new_path)


def contains_wildcard(c):
    i = 0
    n = len(c)
    while i < n:
        if i <= n - 1 and c[i] == '\\' and c[i + 1] in wildcards:
            i += 2
            continue
        elif c[i] in wildcards:
            return True
        i += 1
    return False


def unescape_escaped_wildcards(c):
    new_c = []
    i = 0
    n = len(c)
    while i < n:
        if i <= n - 1 and c[i] == '\\' and c[i + 1] in wildcards:
            new_c.append(c[i + 1])
            i += 2
            continue
        new_c.append(c[i])
        i += 1
    return ''.join(new_c)


def prefix_wout_wildcard(c):
    new_c = []
    i = 0
    n = len(c)
    while i < n:
        if i <= n - 1 and c[i] == '\\' and c[i + 1] in wildcards:
            new_c.append(c[i + 1])
            i += 2
            continue
        elif c[i] in wildcards:
            return ''.join(new_c)
        new_c.append(c[i])
        i += 1
    return ''.join(new_c)


def makedirs(path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)


def cp(src, dest):  # pylint: disable=invalid-name
    src = os.path.abspath(src)
    dest = os.path.abspath(dest)
    makedirs(dest)
    shutil.copy(src, dest)


def listdir(path):
    return os.listdir(path)


def remove(path):
    if os.path.exists(path):
        assert os.path.isfile(path)
        os.remove(path)


def new_file(path, size):
    with open(path, 'ab') as fp:
        fp.truncate(size)


def _glob(path):
    is_dir = path.endswith('/')
    path = os.path.abspath(path)
    if is_dir:
        path += '/'
    paths = glob.glob(escape(path), recursive=True)

    def _listdir(path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if os.path.isfile(path):
            return [(path, os.path.getsize(path))]
        # gsutil doesn't copy empty directories
        return flatten([_listdir(path.rstrip('/') + '/' + f) for f in listdir(path)])

    return flatten([_listdir(path) for path in paths])


class AsyncOS:
    def __init__(self, thread_pool):
        self.thread_pool = thread_pool
        self._wrapped_makedirs = self._wrap(makedirs)
        self._wrapped_cp = self._wrap(cp)
        self._wrapped_listdir = self._wrap(listdir)
        self._wrapped_remove = self._wrap(remove)
        self._wrapped_new_file = self._wrap(new_file)
        self._wrapped_glob = self._wrap(_glob)

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

    async def cp(self, src, dest):  # pylint: disable=invalid-name
        return await self._wrapped_cp(self, src, dest)

    async def listdir(self, path):
        return await self._wrapped_listdir(self, path)

    async def remove(self, path):
        return await self._wrapped_remove(self, path)

    async def new_file(self, path, size):
        return await self._wrapped_new_file(self, path, size)

    async def glob(self, path):
        return await self._wrapped_glob(self, path)
