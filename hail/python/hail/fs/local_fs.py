from typing import Dict, List, BinaryIO
import gzip
import io
import os
import warnings
from stat import S_ISREG, S_ISDIR
from hurry.filesize import size
from shutil import copy2, rmtree

from .fs import FS


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
        try:
            return S_ISREG(os.stat(path).st_mode)
        except FileNotFoundError:
            return False

    def is_dir(self, path: str) -> bool:
        try:
            return self._stat_is_local_dir(os.stat(path))
        except FileNotFoundError:
            return False

    def stat(self, path: str) -> Dict:
        stats = os.stat(path)
        if path[-3:] == '.gz' or path[-4:] == '.bgz':
            warnings.warn('stat on a compressed file requires reading the entire file to report the uncompressed size')
            uncompressed_size_bytes = 0
            with self.open(path, 'rb') as f:
                while True:
                    k = len(f.read(4096))
                    if k == 0:
                        break
                    uncompressed_size_bytes += k
        else:
            uncompressed_size_bytes = stats.st_size
        return {
            'is_dir': self._stat_is_local_dir(stats),
            'size_bytes': uncompressed_size_bytes,
            'size': size(uncompressed_size_bytes),
            'path': path,
            'owner': stats.st_uid,
            'modification_time': stats.st_mtime,
        }

    def _stat_is_local_dir(self, stats: os.stat_result) -> bool:
        return S_ISDIR(stats.st_mode)

    def ls(self, path: str) -> List[Dict]:
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
