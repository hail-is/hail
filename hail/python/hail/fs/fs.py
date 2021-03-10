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
    def mkdir(self, path: str):
        """Ensure files can be created whose dirname is `path`.

        Warning
        -------

        On file systems without a notion of directories, this function will do nothing. For example,
        on Google Cloud Storage, this operation does nothing.

        """
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
