import io
import json
import time
from typing import Any, Dict, List, Union

import dateutil.parser

from hailtop.fs.fs import FS
from hailtop.fs.stat_result import FileListEntry, FileType


def _file_list_entry_scala_to_python(file_list_entry: Dict[str, Any]) -> FileListEntry:
    dt = dateutil.parser.isoparse(file_list_entry['modification_time'])
    mtime = time.mktime(dt.timetuple())
    if file_list_entry['is_dir']:
        typ = FileType.DIRECTORY
    elif file_list_entry['is_link']:
        typ = FileType.SYMLINK
    else:
        typ = FileType.FILE
    return FileListEntry(
        path=file_list_entry['path'],
        owner=file_list_entry['owner'],
        size=file_list_entry['size'],
        typ=typ,
        modification_time=mtime,
    )


class HadoopFS(FS):
    def __init__(self, utils_package_object, jfs):
        self._utils_package_object = utils_package_object
        self._jfs = jfs

    def open(self, path: str, mode: str = 'r', buffer_size: int = 8192):
        return self._open(path, mode, buffer_size, use_codec=False)

    def legacy_open(self, path: str, mode: str = 'r', buffer_size: int = 8192):
        # this method for combatibility with hadoop_open in 0.2
        return self._open(path, mode, buffer_size, use_codec=True)

    def _open(self, path: str, mode: str = 'r', buffer_size: int = 8192, use_codec: bool = False):
        handle: Union[io.BufferedReader, io.BufferedWriter]
        if 'r' in mode:
            handle = io.BufferedReader(
                HadoopReader(self, path, buffer_size, use_codec=use_codec), buffer_size=buffer_size
            )
        elif 'w' in mode:
            handle = io.BufferedWriter(HadoopWriter(self, path, use_codec=use_codec), buffer_size=buffer_size)
        elif 'x' in mode:
            handle = io.BufferedWriter(
                HadoopWriter(self, path, exclusive=True, use_codec=use_codec), buffer_size=buffer_size
            )

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

    def stat(self, path: str) -> FileListEntry:
        stat_dict = json.loads(self._utils_package_object.fileListEntry(self._jfs, path))
        return _file_list_entry_scala_to_python(stat_dict)

    def ls(self, path: str) -> List[FileListEntry]:
        return [
            _file_list_entry_scala_to_python(st) for st in json.loads(self._utils_package_object.ls(self._jfs, path))
        ]

    def mkdir(self, path: str) -> None:
        return self._jfs.mkDir(path)

    def remove(self, path: str):
        return self._jfs.remove(path)

    def rmtree(self, path: str):
        return self._jfs.rmtree(path)

    def supports_scheme(self, scheme: str) -> bool:
        return self._jfs.supportsScheme(scheme)

    def canonicalize_path(self, path: str) -> str:
        return self._jfs.makeQualified(path)


class HadoopReader(io.RawIOBase):
    def __init__(self, hfs, path, buffer_size, use_codec=False):
        super(HadoopReader, self).__init__()
        self._seekable = not use_codec
        if use_codec:
            self._jfile = hfs._utils_package_object.readFileCodec(hfs._jfs, path, buffer_size)
        else:
            self._jfile = hfs._utils_package_object.readFile(hfs._jfs, path, buffer_size)
        self.mode = 'rb'

    def close(self):
        self._jfile.close()

    def readable(self):
        return True

    def seekable(self):
        return self._seekable

    def seek(self, offset, whence=io.SEEK_SET):
        if not 0 <= whence <= 2:
            raise io.UnsupportedOperation(f'unsupported whence value {whence}')
        return self._jfile.seek(offset, whence)

    def tell(self):
        return self._jfile.getPosition()

    def readinto(self, b):
        b_from_java = self._jfile.read(len(b))
        n_read = len(b_from_java)
        b[:n_read] = b_from_java
        return n_read


class HadoopWriter(io.RawIOBase):
    def __init__(self, hfs, path, exclusive=False, use_codec=False):
        super(HadoopWriter, self).__init__()
        if use_codec:
            self._jfile = hfs._utils_package_object.writeFileCodec(hfs._jfs, path, exclusive)
        else:
            self._jfile = hfs._utils_package_object.writeFile(hfs._jfs, path, exclusive)
        if exclusive:
            self.mode = 'xb'
        else:
            self.mode = 'wb'

    def writable(self):
        return True

    def close(self):
        self._jfile.close()

    def flush(self):
        self._jfile.flush()

    def write(self, b):
        self._jfile.write(bytearray(b))
        return len(b)
