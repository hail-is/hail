from typing import Dict, List, Optional
import os
from stat import S_ISREG, S_ISDIR
import gcsfs
from hurry.filesize import size
from shutil import copy2, rmtree
from hailtop.utils import sync_retry_transient_errors

from .fs import FS


class GoogleCloudStorageFS(FS):
    def __init__(self):
        self.client = gcsfs.core.GCSFileSystem(secure_serialize=True)

    def _is_local(self, path: str):
        if path.startswith("gs://"):
            return False
        return True

    def _add_gs_path_prefix(self, path: str) -> str:
        first_idx = 0

        for char in path:
            if char != "/":
                break
            first_idx += 1

        return "gs://" + path[first_idx:]

    def open(self, path: str, mode: str = 'r', buffer_size: int = 2**18):
        if self._is_local(path):
            if mode.startswith('w') and not os.path.exists(path):
                parts = os.path.split(path)
                if not os.path.exists(parts[0]):
                    os.makedirs(parts[0])

            return open(path, mode, buffer_size)

        return self.client.open(path, mode, buffer_size)

    def copy(self, src: str, dest: str):
        src_is_remote = src.startswith('gs://')
        dest_is_remote = dest.startswith('gs://')

        if src_is_remote and dest_is_remote:
            self.client.copy(src, dest)
        elif src_is_remote:
            self.client.get(src, dest)
        elif dest_is_remote:
            self.client.put(src, dest)
        else:
            dst_w_file = dest
            if os.path.isdir(dst_w_file):
                dst_w_file = os.path.join(dest, os.path.basename(src))

            copy2(src, dst_w_file)
            stats = os.stat(src)

            os.chown(dst_w_file, stats.st_uid, stats.st_gid)

    def exists(self, path: str) -> bool:
        if self._is_local(path):
            return os.path.exists(path)

        return self.client.exists(path)

    def is_file(self, path: str) -> bool:
        try:
            if self._is_local(path):
                return S_ISREG(os.stat(path).st_mode)
            return not self._stat_is_gs_dir(self.client.info(path))
        except FileNotFoundError:
            return False

    def is_dir(self, path: str) -> bool:
        try:
            if self._is_local(path):
                return self._stat_is_local_dir(os.stat(path))
            return self._stat_is_gs_dir(self.client.info(path))
        except FileNotFoundError:
            return False

    def stat(self, path: str) -> Dict:
        if self._is_local(path):
            return self._format_stat_local_file(os.stat(path), path)

        return self._format_stat_gs_file(self.client.info(path), path)

    def _format_stat_gs_file(self, stats: Dict, path: Optional[str] = None) -> Dict:
        path_from_stats = stats.get('name')
        if path_from_stats is not None:
            path_from_stats = self._add_gs_path_prefix(path_from_stats)
        else:
            assert path is not None
            path_from_stats = path
        return {
            'is_dir': self._stat_is_gs_dir(stats),
            'size_bytes': stats['size'],
            'size': size(stats['size']),
            'path': path_from_stats,
            'owner': stats['bucket'],
            'modification_time': stats.get('updated')
        }

    def _format_stat_local_file(self, stats: os.stat_result, path: str) -> Dict:
        return {
            'is_dir': self._stat_is_local_dir(stats),
            'size_bytes': stats.st_size,
            'size': size(stats.st_size),
            'path': path,
            'owner': stats.st_uid,
            'modification_time': stats.st_mtime,
        }

    def _stat_is_gs_dir(self, stats: Dict) -> bool:
        return stats['storageClass'] == 'DIRECTORY' or stats['name'].endswith('/')

    def _stat_is_local_dir(self, stats: os.stat_result) -> bool:
        return S_ISDIR(stats.st_mode)

    def ls(self, path: str) -> List[Dict]:
        is_local = self._is_local(path)

        if is_local:
            return [self._format_stat_local_file(os.stat(file), file) for file in os.listdir(path)]

        return [self._format_stat_gs_file(file)
                for file in self.client.ls(path, detail=True)]

    def mkdir(self, path: str):
        pass

    def remove(self, path: str):
        if self._is_local(path):
            os.remove(path)
        self.client.rm(path)

    def rmtree(self, path: str):
        if self._is_local(path):
            rmtree(path)

        def rm_not_exist_ok():
            try:
                self.client.rm(path, recursive=True)
            except FileNotFoundError:
                pass
        sync_retry_transient_errors(rm_not_exist_ok)

    def supports_scheme(self, scheme: str) -> bool:
        return scheme in ("gs", "")
