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
    modification_time: Optional[float]

    def is_dir(self) -> bool:
        return self.typ is FileType.DIRECTORY

    def to_legacy_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'owner': self.owner,
            'is_dir': self.is_dir(),
            'size_bytes': self.size,
            'size': hurry.filesize.size(self.size),
            'modification_time': self.modification_time,
        }
