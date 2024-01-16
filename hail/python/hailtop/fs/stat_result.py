from enum import Enum, auto
from typing import Any, Dict, NamedTuple, Optional, Union

from hailtop.utils.filesize import filesize


class FileType(Enum):
    DIRECTORY = auto()
    FILE = auto()
    SYMLINK = auto()


class FileStatus(NamedTuple):
    path: str
    owner: Union[None, str, int]
    size: int
    # common point between unix, google, and hadoop filesystems, represented as a unix timestamp
    modification_time: Optional[float]

    def to_legacy_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'owner': self.owner,
            'size_bytes': self.size,
            'size': filesize(self.size),
            'modification_time': self.modification_time,
        }


class FileListEntry(NamedTuple):
    path: str
    owner: Union[None, str, int]
    size: int
    typ: FileType
    # common point between unix, google, and hadoop filesystems, represented as a unix timestamp
    modification_time: Optional[float]

    def is_dir(self) -> bool:
        return self.typ is FileType.DIRECTORY

    def to_legacy_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'owner': self.owner,
            'size_bytes': self.size,
            'size': filesize(self.size),
            'modification_time': self.modification_time,
            'is_dir': self.is_dir(),
        }


StatResult = FileListEntry  # backwards compatibility
