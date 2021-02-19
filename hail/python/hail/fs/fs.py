import abc
import sys
import os
from typing import Dict, List

from hail.utils.java import Env, info
from hail.utils import local_path_uri


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

    @abc.abstractmethod
    def remove(self, path: str):
        pass

    @abc.abstractmethod
    def rmtree(self, path: str):
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


class DeletingFile:
    def __init__(self, fs: FS, fname: str):
        self.fs = fs
        self.fname = fname

    def __enter__(self):
        return self.fname

    def __exit__(self, type, value, traceback):
        return self.fs.remove(self.fname)


class DeletingDirectory:
    def __init__(self, fs: FS, dirname: str):
        self.fs = fs
        self.dirname = dirname

    def __enter__(self):
        return self.dirname

    def __exit__(self, type, value, traceback):
        return self.fs.rmtree(self.dirname)
