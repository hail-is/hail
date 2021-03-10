import io
import json
from typing import Dict, List

from .fs import FS


class HadoopFS(FS):
    def __init__(self, utils_package_object, jfs):
        self._utils_package_object = utils_package_object
        self._jfs = jfs

    def open(self, path: str, mode: str = 'r', buffer_size: int = 8192):
        if 'r' in mode:
            handle = io.BufferedReader(HadoopReader(self, path, buffer_size), buffer_size=buffer_size)
        elif 'w' in mode:
            handle = io.BufferedWriter(HadoopWriter(self, path), buffer_size=buffer_size)
        elif 'x' in mode:
            handle = io.BufferedWriter(HadoopWriter(self, path, exclusive=True), buffer_size=buffer_size)

        if 'b' in mode:
            return handle
        else:
            return io.TextIOWrapper(handle, encoding='iso-8859-1')

    def copy(self, src: str, dest: str):
        self._jfs.copy(src, dest, False)

    def exists(self, path: str) -> bool:
        return self._jfs.exists(path)

    def is_file(self, path: str) -> bool:
        return self._jfs.isFile(path)

    def is_dir(self, path: str) -> bool:
        return self._jfs.isDir(path)

    def stat(self, path: str) -> Dict:
        return json.loads(self._utils_package_object.stat(self._jfs, path))

    def ls(self, path: str) -> List[Dict]:
        return json.loads(self._utils_package_object.ls(self._jfs, path))

    def mkdir(self, path: str) -> None:
        return self._jfs.mkDir(path)

    def remove(self, path: str):
        return self._jfs.remove(path)

    def rmtree(self, path: str):
        return self._jfs.rmtree(path)


class HadoopReader(io.RawIOBase):
    def __init__(self, hfs, path, buffer_size):
        super(HadoopReader, self).__init__()
        self._jfile = hfs._utils_package_object.readFile(hfs._jfs, path, buffer_size)

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
    def __init__(self, hfs, path, exclusive=False):
        self._jfile = hfs._utils_package_object.writeFile(hfs._jfs, path, exclusive)
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
