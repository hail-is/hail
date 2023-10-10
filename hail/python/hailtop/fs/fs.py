import abc
from typing import IO, List

from .stat_result import FileListEntry


class FS(abc.ABC):
    @abc.abstractmethod
    def open(self, path: str, mode: str = 'r', buffer_size: int = 8192) -> IO:
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self, src: str, dest: str):
        raise NotImplementedError

    @abc.abstractmethod
    def exists(self, path: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def is_file(self, path: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def is_dir(self, path: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def stat(self, path: str) -> FileListEntry:
        raise NotImplementedError

    @abc.abstractmethod
    def ls(self, path: str) -> List[FileListEntry]:
        raise NotImplementedError

    @abc.abstractmethod
    def mkdir(self, path: str):
        raise NotImplementedError

    @abc.abstractmethod
    def remove(self, path: str):
        raise NotImplementedError

    @abc.abstractmethod
    def rmtree(self, path: str):
        raise NotImplementedError

    @abc.abstractmethod
    def supports_scheme(self, scheme: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def canonicalize_path(self, path: str) -> str:
        raise NotImplementedError
