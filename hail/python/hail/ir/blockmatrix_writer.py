import abc
import json

from ..typecheck import typecheck_method, sequenceof, nullable, enumeration
from ..expr.types import tvoid, tstr
from ..utils.misc import escape_str


class BlockMatrixWriter(object):
    @abc.abstractmethod
    def render(self):
        pass

    @abc.abstractmethod
    def _type(self):
        pass

    @abc.abstractmethod
    def __eq__(self, other):
        pass


class BlockMatrixNativeWriter(BlockMatrixWriter):
    @typecheck_method(path=str, overwrite=bool, force_row_major=bool, stage_locally=bool)
    def __init__(self, path, overwrite, force_row_major, stage_locally):
        self.path = path
        self.overwrite = overwrite
        self.force_row_major = force_row_major
        self.stage_locally = stage_locally

    def render(self):
        writer = {'name': 'BlockMatrixNativeWriter',
                  'path': self.path,
                  'overwrite': self.overwrite,
                  'forceRowMajor': self.force_row_major,
                  'stageLocally': self.stage_locally}
        return escape_str(json.dumps(writer))

    def _type(self):
        return tvoid

    def __eq__(self, other):
        return isinstance(other, BlockMatrixNativeWriter) and \
            self.path == other.path and \
            self.overwrite == other.overwrite and \
            self.force_row_major == other.force_row_major and \
            self.stage_locally == other.stage_locally


class BlockMatrixBinaryWriter(BlockMatrixWriter):
    @typecheck_method(path=str)
    def __init__(self, path):
        self.path = path

    def render(self):
        writer = {'name': 'BlockMatrixBinaryWriter',
                  'path': self.path}
        return escape_str(json.dumps(writer))

    def _type(self):
        return tstr

    def __eq__(self, other):
        return isinstance(other, BlockMatrixBinaryWriter) and \
            self.path == other.path


class BlockMatrixRectanglesWriter(BlockMatrixWriter):
    @typecheck_method(path=str,
                      rectangles=sequenceof(sequenceof(int)),
                      delimiter=str,
                      binary=bool)
    def __init__(self, path, rectangles, delimiter, binary):
        self.path = path
        self.rectangles = rectangles
        self.delimiter = delimiter
        self.binary = binary

    def render(self):
        writer = {'name': 'BlockMatrixRectanglesWriter',
                  'path': self.path,
                  'rectangles': self.rectangles,
                  'delimiter': self.delimiter,
                  'binary': self.binary}
        return escape_str(json.dumps(writer))

    def _type(self):
        return tvoid

    def __eq__(self, other):
        return isinstance(other, BlockMatrixRectanglesWriter) and \
            self.path == other.path and \
            self.rectangles == other.rectangles and \
            self.delimiter == other.delimiter and \
            self.binary == other.binary


class BlockMatrixMultiWriter(object):
    @abc.abstractmethod
    def render(self):
        pass

    def _type(self):
        return tvoid

    @abc.abstractmethod
    def __eq__(self, other):
        pass


class BlockMatrixBinaryMultiWriter(BlockMatrixMultiWriter):
    @typecheck_method(prefix=str, overwrite=bool)
    def __init__(self, prefix, overwrite):
        self.prefix = prefix
        self.overwrite = overwrite

    def render(self):
        writer = {'name': 'BlockMatrixBinaryMultiWriter',
                  'prefix': self.prefix,
                  'overwrite': self.overwrite}
        return escape_str(json.dumps(writer))

    def _type(self):
        return tvoid

    def __eq__(self, other):
        return isinstance(other, BlockMatrixBinaryMultiWriter) and \
            self.prefix == other.prefix and \
            self.overwrite == other.overwrite


class BlockMatrixTextMultiWriter(BlockMatrixMultiWriter):
    @typecheck_method(prefix=str, overwrite=bool, delimiter=str, header=nullable(str), add_index=bool,
                      compression=nullable(enumeration('gz', 'bgz')), custom_filenames=nullable(sequenceof(str)))
    def __init__(self, prefix, overwrite, delimiter, header, add_index, compression, custom_filenames):
        self.prefix = prefix
        self.overwrite = overwrite
        self.delimiter = delimiter
        self.header = header
        self.add_index = add_index
        self.compression = compression
        self.custom_filenames = custom_filenames

    def render(self):
        writer = {'name': 'BlockMatrixTextMultiWriter',
                  'prefix': self.prefix,
                  'overwrite': self.overwrite,
                  'delimiter': self.delimiter,
                  'header': self.header,
                  'addIndex': self.add_index,
                  'compression': self.compression,
                  'customFilenames': self.custom_filenames}
        return escape_str(json.dumps(writer))

    def _type(self):
        return tvoid

    def __eq__(self, other):
        return isinstance(other, BlockMatrixTextMultiWriter) and \
            self.prefix == other.prefix and \
            self.overwrite == other.overwrite and \
            self.delimiter == other.overwrite and \
            self.header == other.header and \
            self.add_index == other.add_index and \
            self.compression == other.compression and \
            self.custom_filenames == other.custom_filenames


class BlockMatrixPersistWriter(BlockMatrixWriter):
    @typecheck_method(id=str, storage_level=str)
    def __init__(self, id, storage_level):
        self.id = id
        self.storage_level = storage_level

    def render(self):
        writer = {'name': 'BlockMatrixPersistWriter',
                  'id': self.id,
                  'storageLevel': self.storage_level}
        return escape_str(json.dumps(writer))

    def _type(self):
        return tvoid

    def __eq__(self, other):
        return isinstance(other, BlockMatrixPersistWriter) and \
            self.id == other.id and \
            self.storage_level == other.storage_level


class BlockMatrixNativeMultiWriter(BlockMatrixMultiWriter):
    @typecheck_method(prefix=str, overwrite=bool, force_row_major=bool, stage_locally=bool)
    def __init__(self, prefix, overwrite, force_row_major, stage_locally):
        self.prefix = prefix
        self.overwrite = overwrite
        self.force_row_major = force_row_major
        self.stage_locally = stage_locally

    def render(self):
        writer = {'name': 'BlockMatrixNativeMultiWriter',
                  'prefix': self.prefix,
                  'overwrite': self.overwrite,
                  'forceRowMajor': self.force_row_major,
                  'stageLocally': self.stage_locally}
        return escape_str(json.dumps(writer))

    def _type(self):
        return tvoid

    def __eq__(self, other):
        return isinstance(other, BlockMatrixNativeMultiWriter) and \
            self.prefix == other.prefix and \
            self.overwrite == other.overwrite and \
            self.force_row_major == other.force_row_major and \
            self.stage_locally == other.stage_locally
