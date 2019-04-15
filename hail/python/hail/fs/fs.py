import abc
from hail.utils.java import Env, info
from hail.utils import local_path_uri
import io
import json
from typing import Dict, List
import sys
import os
from google.oauth2 import service_account
import gcsfs


class FS(abc.ABC):
    @abc.abstractmethod
    def open(self, path: str, mode: str = 'r', buffer_size: int = 8192):
        pass

    @abc.abstractmethod
    def copy(self, src: str, dest: str):
        pass

    @abc.abstractmethod
    def exists(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def is_file(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def is_dir(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def stat(self, path: str) -> Dict:
        pass

    @abc.abstractmethod
    def ls(self, path: str) -> List[Dict]:
        pass

    def copy_log(self, path: str) -> None:
        log = Env.hc()._log
        try:
            if self.is_dir(path):
                _, tail = os.path.split(log)
                path = os.path.join(path, tail)
            info(f"copying log to {repr(path)}...")
            self.copy(local_path_uri(Env.hc()._log), path)
        except Exception as e:
            sys.stderr.write(f'Could not copy log: encountered error:\n  {e}')


class HadoopFS(FS):
    def open(self, path: str, mode: str = 'r', buffer_size: int = 8192):
        if 'r' in mode:
            handle = io.BufferedReader(HadoopReader(path, buffer_size), buffer_size=buffer_size)
        elif 'w' in mode:
            handle = io.BufferedWriter(HadoopWriter(path), buffer_size=buffer_size)
        elif 'x' in mode:
            handle = io.BufferedWriter(HadoopWriter(path, exclusive=True), buffer_size=buffer_size)

        if 'b' in mode:
            return handle
        else:
            return io.TextIOWrapper(handle, encoding='iso-8859-1')

    def copy(self, src: str, dest: str):
        Env.jutils().copyFile(src, dest, Env.hc()._jhc)

    def exists(self, path: str) -> bool:
        return Env.jutils().exists(path, Env.hc()._jhc)

    def is_file(self, path: str) -> bool:
        return Env.jutils().isFile(path, Env.hc()._jhc)

    def is_dir(self, path: str) -> bool:
        return Env.jutils().isDir(path, Env.hc()._jhc)

    def stat(self, path: str) -> Dict:
        return json.loads(Env.jutils().stat(path, Env.hc()._jhc))

    def ls(self, path: str) -> List[Dict]:
        r = Env.jutils().ls(path, Env.hc()._jhc)
        return json.loads(r)


class HadoopReader(io.RawIOBase):
    def __init__(self, path, buffer_size):
        self._jfile = Env.jutils().readFile(path, Env.hc()._jhc, buffer_size)
        super(HadoopReader, self).__init__()

    def close(self):
        self._jfile.close()

    def readable(self):
        return True

    def readinto(self, b):
        b_from_java = self._jfile.read(len(b))
        n_read = len(b_from_java)
        b[:n_read] = b_from_java
        return n_read


class HadoopWriter(io.RawIOBase):
    def __init__(self, path, exclusive=False):
        self._jfile = Env.jutils().writeFile(path, Env.hc()._jhc, exclusive)
        super(HadoopWriter, self).__init__()

    def writable(self):
        return True

    def close(self):
        self._jfile.close()

    def flush(self):
        self._jfile.flush()

    def write(self, b):
        self._jfile.write(bytearray(b))
        return len(b)


class GoogleCloudStorageFS(FS):
    def __init__(self):
        credentials = service_account.Credentials.from_service_account_file(
            filename='/gsa-key/privateKeyData',
            scopes=['https://www.googleapis.com/auth/cloud-platform'])

        self.client = gcsfs.core.GCSFileSystem(credentials, secure_serialize=True)

    def open(self, path: str, mode: str = 'r'):
        return self.client.open(path, mode)

    def copy(self, src: str, dest: str):
        if src.startswith('gs://'):
            return self.client.copy(src, dest)
        else:
            return self.client.put(src, dest)

    def exists(self, path: str) -> bool:
        return self.client.exists(path)

    def is_file(self, path: str) -> bool:
        stats = self.client.info(path)

        return not self._stat_is_dir(stats)

    def is_dir(self, path: str) -> bool:
        stats = self.client.info(path)

        return self._stat_is_dir(stats)

    def stat(self, path: str) -> Dict:
        stats = self.client.info(path)

        return {
            'is_dir': self._stat_is_dir(stats),
            'size_bytes': stats['size'],
            'size': stats['size'],
            'path': stats['path'],
            'owner': stats['bucket'],
            'modification_time': stats.get('updated')
        }

    def _stat_is_dir(self, stats: Dict):
        return stats['storageClass'] == 'DIRECTORY' or stats['name'].endswith('/')

    def ls(self, path: str) -> List[Dict]:
        files = self.client.ls(path)

        return [self.stat(file) for file in files]
