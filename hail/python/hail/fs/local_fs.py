from typing import List, BinaryIO
import gzip
import io
import os
from shutil import copy2, rmtree

from .fs import FS
from .stat_result import StatResult


class LocalFS(FS):
    def __init__(self):
        pass

    def open(self, path: str, mode: str = 'r', buffer_size: int = 0):
        if mode not in ('r', 'rb', 'w', 'wb'):
            raise ValueError(f'Unsupported mode: {repr(mode)}')

        strm: BinaryIO
        if mode[0] == 'r':
            strm = open(path, 'rb')
        else:
            assert mode[0] == 'w'
            strm = open(path, 'wb')

        if path[-3:] == '.gz' or path[-4:] == '.bgz':
            strm = gzip.GzipFile(fileobj=strm, mode=mode)  # type: ignore # GzipFile should be a BinaryIO
        if 'b' not in mode:
            strm = io.TextIOWrapper(strm, encoding='utf-8')  # type: ignore # TextIOWrapper should be a BinaryIO
        return strm

    def copy(self, src: str, dest: str):
        dst_w_file = dest
        if os.path.isdir(dst_w_file):
            dst_w_file = os.path.join(dest, os.path.basename(src))

        copy2(src, dst_w_file)
        stats = os.stat(src)

        os.chown(dst_w_file, stats.st_uid, stats.st_gid)

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def is_file(self, path: str) -> bool:
        return os.path.isfile(path)

    def is_dir(self, path: str) -> bool:
        return os.path.isdir(path)

    def stat(self, path: str) -> StatResult:
        return StatResult.from_os_stat_result(path, os.stat(path))

    def ls(self, path: str) -> List[StatResult]:
        return [self.stat(os.path.join(path, file))
                for file in os.listdir(path)]

    def mkdir(self, path: str):
        os.mkdir(path)

    def remove(self, path: str):
        os.remove(path)

    def rmtree(self, path: str):
        rmtree(path)

    def supports_scheme(self, scheme: str) -> bool:
        return scheme == ""
