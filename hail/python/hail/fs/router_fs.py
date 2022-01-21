from typing import Dict, List, Any, AsyncContextManager, BinaryIO
import asyncio
import gzip
import hurry
import io
import nest_asyncio

from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.aiotools.fs import Copier, Transfer, FileListEntry, ReadableStream, WritableStream
from hailtop.utils import async_to_blocking, OnlineBoundedGather2


from .fs import FS


class SyncReadableStream(io.RawIOBase, BinaryIO):  # type: ignore # https://github.com/python/typeshed/blob/a40d79a4e63c4e750a8d3a8012305da942251eb4/stdlib/http/client.pyi#L81
    def __init__(self, ars: ReadableStream, name: str):
        super().__init__()
        self.ars = ars
        self._mode = 'rb'
        self._name = name

    @property
    def mode(self):
        return self._mode

    @property
    def name(self):
        return self._name

    def close(self):
        self.ars.close()
        async_to_blocking(self.ars.wait_closed())

    @property
    def closed(self) -> bool:
        return self.ars.closed

    def fileno(self) -> int:
        raise OSError

    def flush(self):
        pass

    def isatty(self):
        return False

    def readable(self):
        return True

    def seek(self, offset, whence=None):
        raise OSError

    def seekable(self):
        return False

    def tell(self):
        raise io.UnsupportedOperation

    def truncate(self):
        raise io.UnsupportedOperation

    def writable(self):
        return False

    def writelines(self, lines):
        raise OSError

    def read(self, size=-1) -> bytes:
        return async_to_blocking(self.ars.read(size))

    def readall(self) -> bytes:
        return async_to_blocking(self.ars.read(-1))

    def readinto(self, b):
        b[:] = async_to_blocking(self.ars.readexactly(len(b)))

    def write(self, b):
        raise OSError


class SyncWritableStream(io.RawIOBase, BinaryIO):  # type: ignore # https://github.com/python/typeshed/blob/a40d79a4e63c4e750a8d3a8012305da942251eb4/stdlib/http/client.pyi#L81
    def __init__(self, cm: AsyncContextManager[WritableStream], name: str):
        super().__init__()
        self.cm = cm
        self.aws = async_to_blocking(self.cm.__aenter__())
        self._mode = 'wb'
        self._name = name

    @property
    def mode(self):
        return self._mode

    @property
    def name(self):
        return self._name

    def close(self):
        self.aws.close()
        async_to_blocking(self.cm.__aexit__())

    @property
    def closed(self) -> bool:
        return self.aws.closed

    def fileno(self) -> int:
        raise OSError

    def flush(self):
        pass

    def isatty(self):
        return False

    def readable(self):
        return False

    def readline(self, size=-1):
        raise OSError

    def readlines(self, hint=-1):
        raise OSError

    def seek(self, offset, whence=None):
        raise OSError

    def seekable(self):
        return False

    def tell(self):
        raise io.UnsupportedOperation

    def truncate(self):
        raise io.UnsupportedOperation

    def writable(self):
        return True

    def read(self, size=-1):
        raise OSError

    def readall(self):
        raise OSError

    def readinto(self, b):
        raise OSError

    def write(self, b):
        return async_to_blocking(self.aws.write(b))


def _stat_dict(is_dir: bool, size_bytes: int, path: str) -> Dict[str, Any]:
    return {
        'is_dir': is_dir,
        'size_bytes': size_bytes,
        'size': hurry.filesize.size(size_bytes),
        'path': path,
    }


class RouterFS(FS):
    def __init__(self, afs: RouterAsyncFS):
        nest_asyncio.apply()
        self.afs = afs

    def open(self, path: str, mode: str = 'r', buffer_size: int = 8192) -> io.IOBase:
        del buffer_size

        if mode not in ('r', 'rb', 'w', 'wb'):
            raise ValueError(f'Unsupported mode: {repr(mode)}')

        strm: io.IOBase
        if mode[0] == 'r':
            strm = SyncReadableStream(async_to_blocking(self.afs.open(path)), path)
        else:
            assert mode[0] == 'w'
            strm = SyncWritableStream(async_to_blocking(self.afs.create(path)), path)

        if path[-3:] == '.gz' or path[-4:] == '.bgz':
            strm = gzip.GzipFile(fileobj=strm, mode=mode)
        if 'b' not in mode:
            strm = io.TextIOWrapper(strm, encoding='utf-8')  # type: ignore # typeshed is wrong, this *is* an IOBase
        return strm

    def copy(self, src: str, dest: str, *, max_simultaneous_transfers=75):
        transfer = Transfer(src, dest)

        async def _copy():
            sema = asyncio.Semaphore(max_simultaneous_transfers)
            async with sema:
                await Copier.copy(self.afs, asyncio.Semaphore, transfer)
        return async_to_blocking(_copy())

    def exists(self, path: str) -> bool:
        return async_to_blocking(self.afs.exists(path))

    def is_file(self, path: str) -> bool:
        return async_to_blocking(self.afs.isfile(path))

    async def _async_is_dir(self, path: str) -> bool:
        if path[-1] != '/':
            path = path + '/'
        return await self.afs.isdir(path)

    def is_dir(self, path: str) -> bool:
        return async_to_blocking(self._async_is_dir(path))

    def stat(self, path: str) -> Dict:
        async def size_bytes_or_none():
            try:
                return await (await self.afs.statfile(path)).size()
            except FileNotFoundError:
                return None
        size_bytes, is_dir = async_to_blocking(asyncio.gather(
            size_bytes_or_none(), self._async_is_dir(path)))
        if size_bytes is None:
            if not is_dir:
                raise FileNotFoundError(path)
            return _stat_dict(True, 0, path)
        return _stat_dict(is_dir, size_bytes, path)

    async def _fle_to_dict(self, fle: FileListEntry) -> Dict[str, Any]:
        async def size():
            try:
                return await (await fle.status()).size()
            except IsADirectoryError:
                return 0
        return _stat_dict(
            *await asyncio.gather(fle.is_dir(), size(), fle.url()))

    def ls(self, path: str, _max_simultaneous_files: int = 50) -> List[Dict]:
        async def _ls():
            async with OnlineBoundedGather2(asyncio.Semaphore(_max_simultaneous_files)) as pool:
                tasks = [pool.call(self._fle_to_dict, fle)
                         async for fle in await self.afs.listfiles(path)]
                return [await t for t in tasks]
        return async_to_blocking(_ls())

    def mkdir(self, path: str):
        return async_to_blocking(self.afs.mkdir(path))

    def remove(self, path: str):
        return async_to_blocking(self.remove(path))

    def rmtree(self, path: str):
        return async_to_blocking(self.afs.rmtree(None, path))

    def supports_scheme(self, scheme: str) -> bool:
        return scheme in self.afs.schemes
