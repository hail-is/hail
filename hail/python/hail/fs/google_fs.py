import os
from typing import Dict, List
import gcsfs
from hurry.filesize import size

from .fs import FS


class GoogleCloudStorageFS(FS):
    def __init__(self):
        if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/gsa-key/privateKeyData'

        self.client = gcsfs.core.GCSFileSystem(secure_serialize=True)

    def open(self, path: str, mode: str = 'r', buffer_size: int = 2**18):
        return self.client.open(path, mode, buffer_size)

    def copy(self, src: str, dest: str):
        if src.startswith('gs://'):
            return self.client.copy(src, dest)
        else:
            return self.client.put(src, dest)

    def exists(self, path: str) -> bool:
        return self.client.exists(path)

    def is_file(self, path: str) -> bool:
        try:
            return not self._stat_is_dir(self.client.info(path))
        except FileNotFoundError:
            return False

    def is_dir(self, path: str) -> bool:
        try:
            return self._stat_is_dir(self.client.info(path))
        except FileNotFoundError:
            return False

    def stat(self, path: str) -> Dict:
        return self._process_obj(self.client.info(path))

    def _process_obj(self, stats: Dict) -> Dict:
        return {
            'is_dir': self._stat_is_dir(stats),
            'size_bytes': stats['size'],
            'size': size(stats['size']),
            'path': stats['path'],
            'owner': stats['bucket'],
            'modification_time': stats.get('updated')
        }

    def _stat_is_dir(self, stats: Dict) -> bool:
        return stats['storageClass'] == 'DIRECTORY' or stats['name'].endswith('/')

    def ls(self, path: str) -> List[Dict]:
        files = self.client.ls(path, detail=True)

        return [self._process_obj(file) for file in files]
