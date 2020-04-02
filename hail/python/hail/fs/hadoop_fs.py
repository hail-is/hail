import io
import json
from typing import Dict, List

from .fs import FS
from hail.utils.java import Env


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
        Env.jutils().copyFile(src, dest, Env.backend()._jhc)

    def exists(self, path: str) -> bool:
        return Env.jutils().exists(path, Env.backend()._jhc)

    def is_file(self, path: str) -> bool:
        return Env.jutils().isFile(path, Env.backend()._jhc)

    def is_dir(self, path: str) -> bool:
        return Env.jutils().isDir(path, Env.backend()._jhc)

    def stat(self, path: str) -> Dict:
        return json.loads(Env.jutils().stat(path, Env.backend()._jhc))

    def ls(self, path: str) -> List[Dict]:
        r = Env.jutils().ls(path, Env.backend()._jhc)
        return json.loads(r)


class HadoopReader(io.RawIOBase):
    def __init__(self, path, buffer_size):
        self._jfile = Env.jutils().readFile(path, Env.backend()._jhc, buffer_size)
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
        self._jfile = Env.jutils().writeFile(path, Env.backend()._jhc, exclusive)
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
