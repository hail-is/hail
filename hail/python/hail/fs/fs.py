import abc
import sys
import os
from typing import Dict, List


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
