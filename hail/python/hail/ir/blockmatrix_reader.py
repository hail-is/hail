import abc
import json

from ..typecheck import *
from ..utils.java import escape_str


class BlockMatrixReader(object):
    @abc.abstractmethod
    def render(self):
        pass

    @abc.abstractmethod
    def __eq__(self, other):
        pass


class BlockMatrixNativeReader(BlockMatrixReader):
    @typecheck_method(path=str)
    def __init__(self, path):
        self.path = path

    def render(self):
        reader = {'name': 'BlockMatrixNativeReader',
                  'path': self.path}
        return escape_str(json.dumps(reader))

    def __eq__(self, other):
        return isinstance(other, BlockMatrixNativeReader) and \
               self.path == other.path


class BlockMatrixBinaryReader(BlockMatrixReader):
    @typecheck_method(path=str, shape=sequenceof(int), block_size=int)
    def __init__(self, path, shape, block_size):
        self.path = path
        self.shape = shape
        self.block_size = block_size

    def render(self):
        reader = {'name': 'BlockMatrixBinaryReader',
                  'path': self.path,
                  'shape': self.shape,
                  'blockSize': self.block_size}
        return escape_str(json.dumps(reader))

    def __eq__(self, other):
        return isinstance(other, BlockMatrixBinaryReader) and \
               self.path == other.path and \
               self.shape == other.shape and \
               self.block_size == other.block_size
