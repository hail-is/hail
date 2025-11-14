from enum import Enum, auto
from typing import Any, NamedTuple

from hailtop.utils.filesize import filesize


class FileType(Enum):
    DIRECTORY = auto()
    FILE = auto()
    SYMLINK = auto()


class FileStatus(NamedTuple):
    path: str
    owner: str | int | None
    size: int
    # common point between unix, google, and hadoop filesystems, represented as a unix timestamp
    modification_time: float | None

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            'path': self.path,
            'owner': self.owner,
            'size_bytes': self.size,
            'size': filesize(self.size),
            'modification_time': self.modification_time,
        }


class FileListEntry(NamedTuple):
    """Data returned by :func:`hailtop.fs.stat` or :func:`hailtop.fs.ls`"""

    path: str
    owner: str | int | None
    """Optional name or id of the entry's owner"""
    size: int
    """Size in bytes"""
    typ: FileType
    # common point between unix, google, and hadoop filesystems, represented as a unix timestamp
    modification_time: float | None
    """An optional floating point unix timestamp in seconds"""

    def is_dir(self) -> bool:
        return self.typ is FileType.DIRECTORY

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            'path': self.path,
            'owner': self.owner,
            'size_bytes': self.size,
            'size': filesize(self.size),
            'modification_time': self.modification_time,
            'is_dir': self.is_dir(),
        }


StatResult = FileListEntry  # backwards compatibility
