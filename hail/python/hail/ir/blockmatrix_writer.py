import abc
import json

from ..typecheck import *
from ..utils.java import escape_str


class BlockMatrixWriter(object):
    @abc.abstractmethod
    def render(self, r):
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

    def render(self, r):
        writer = {'name': 'BlockMatrixNativeWriter',
                  'path': self.path,
                  'overwrite': self.overwrite,
                  'forceRowMajor': self.force_row_major,
                  'stageLocally': self.stage_locally}
        return escape_str(json.dumps(writer))

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

    def render(self, r):
        writer = {'name': 'BlockMatrixBinaryWriter',
                  'path': self.path}
        return escape_str(json.dumps(writer))

    def __eq__(self, other):
        return isinstance(other, BlockMatrixBinaryWriter) and \
               self.path == other.path
