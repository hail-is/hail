import os
from stat import S_ISREG, S_ISDIR
from typing import Dict, List
from hurry.filesize import size
from shutil import copy2, rmtree

from .fs import FS


class LocalFS(FS):
    def __init__(self):
        pass

    def open(self, path: str, mode: str = 'r', buffer_size: int = 0):
        return open(path, mode)

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
        return self._format_stat_local_file(os.stat(path), path)

    def _format_stat_local_file(self, stats: os.stat_result, path: str) -> Dict:
        return {
            'is_dir': self._stat_is_local_dir(stats),
            'size_bytes': stats.st_size,
            'size': size(stats.st_size),
            'path': path,
            'owner': stats.st_uid,
            'modification_time': stats.st_mtime,
        }

    def _stat_is_local_dir(self, stats: os.stat_result) -> bool:
        return S_ISDIR(stats.st_mode)

    def ls(self, path: str) -> List[Dict]:
        return [self._format_stat_local_file(os.stat(file), file) for file in os.listdir(path)]

    def mkdir(self, path: str):
        os.mkdir(path)

    def remove(self, path: str):
        os.remove(path)

    def rmtree(self, path: str):
        rmtree(path)
