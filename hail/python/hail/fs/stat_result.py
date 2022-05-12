import os
import stat

from enum import Enum, auto
from typing import Dict, NamedTuple, Optional, Union, Any

import hurry.filesize


class FileType(Enum):
    DIRECTORY = auto()
    FILE = auto()
    SYMLINK = auto()


class StatResult(NamedTuple):
    path: str
    owner: Union[None, str, int]
    size: int
    typ: FileType
    # common point between unix, google, and hadoop filesystems, represented as a unix timestamp
    modification_time: Optional[float] = None

    def is_dir(self) -> bool:
        return self.typ is FileType.DIRECTORY

    @staticmethod
    def from_os_stat_result(path: str, sb: os.stat_result) -> 'StatResult':
        if stat.S_ISDIR(sb.st_mode):
            typ = FileType.DIRECTORY
        elif stat.S_ISLNK(sb.st_mode):
            typ = FileType.SYMLINK
        else:
            typ = FileType.FILE
        return StatResult(path=path, owner=sb.st_uid, size=sb.st_size, typ=typ,
                          modification_time=sb.st_mtime)

    def to_legacy_dict(self) -> Dict[str, Any]:
        return dict(path=self.path, owner=self.owner, is_dir=self.is_dir(), size_bytes=self.size,
                    size=hurry.filesize.size(self.size), modification_time=self.modification_time)
