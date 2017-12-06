from hail.java import Env
from hail.java import handle_py4j, scala_object
from hail.typecheck import *

block_matrix_type = lazy()

class BlockMatrix(object):
    @staticmethod
    @handle_py4j
    def read(path):
        hc = Env.hc()
        return BlockMatrix(hc,
            Env.hail().distributedmatrix.BlockMatrix.read(
                hc._jhc, path))

    @staticmethod
    @handle_py4j
    def random(rows, cols, block_size):
        hc = Env.hc()
        return BlockMatrix(hc,
            scala_object(Env.hail().distributedmatrix, 'BlockMatrix').random(
                hc._jhc, rows, cols, block_size))

    def __init__(self, hc, jbm):
        self.hc = hc
        self._jbm = jbm

    @property
    @handle_py4j
    def num_rows(self):
        return self._jbm.rows()
        
    @property
    @handle_py4j
    def num_columns(self):
        return self._jbm.cols()

    @property
    @handle_py4j
    def block_size(self):
        return self._jbm.blockSize()

    @handle_py4j
    @typecheck_method(path=strlike)
    def write(self, path):
        self._jbm.write(path)

    @handle_py4j
    @typecheck_method(that=block_matrix_type)
    def __mul__(self, that):
        return BlockMatrix(self.hc,
            self._jbm.multiply(that._jbm))

block_matrix_type.set(BlockMatrix)
