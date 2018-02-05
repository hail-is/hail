from hail.utils import new_temp_file, storage_level
from hail.utils.java import Env, handle_py4j, scala_object, jarray, numpy_from_breeze
from hail.typecheck import *
from hail.api2 import MatrixTable
from hail.expr.expression import expr_numeric, to_expr, analyze

block_matrix_type = lazy()

class BlockMatrix(object):

    @classmethod
    @handle_py4j
    def default_block_size(cls):
        return scala_object(Env.hail().distributedmatrix, "BlockMatrix").defaultBlockSize()

    @classmethod
    @handle_py4j
    def read(cls, path):
        hc = Env.hc()
        return cls(Env.hail().distributedmatrix.BlockMatrix.read(
            hc._jhc, path))

    @classmethod
    @handle_py4j
    @typecheck_method(entry_expr=expr_numeric,
                      path=nullable(strlike),
                      block_size=nullable(integral))
    def from_matrix_table(cls, entry_expr, path=None, block_size=None):
        if not path:
            path = new_temp_file(suffix="bm")
        if not block_size:
            block_size = cls.default_block_size()
        source = entry_expr._indices.source
        if not isinstance(source, MatrixTable):
            raise ValueError("Expect an expression of 'MatrixTable', found {}".format(
                "expression of '{}'".format(source.__class__) if source is not None else 'scalar expression'))
        mt = source
        base, _ = mt._process_joins(entry_expr)
        analyze('block_matrix_from_expr', entry_expr, mt._entry_indices)

        mt._jvds.writeBlockMatrix(path, to_expr(entry_expr)._ast.to_hql(), block_size)
        return cls.read(path)

    @staticmethod
    @handle_py4j
    def random(num_rows, num_cols, block_size, seed=0, gaussian=False):
        hc = Env.hc()
        return BlockMatrix(scala_object(Env.hail().distributedmatrix, 'BlockMatrix').random(
                hc._jhc, num_rows, num_cols, block_size, seed, gaussian))

    def __init__(self, jbm):
        self._jbm = jbm

    @property
    @handle_py4j
    def num_rows(self):
        return self._jbm.nRows()

    @property
    @handle_py4j
    def num_cols(self):
        return self._jbm.nCols()

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
    def dot(self, that):
        return BlockMatrix(self._jbm.multiply(that._jbm))

    @handle_py4j
    @typecheck_method(cols_to_keep=listof(integral))
    def filter_cols(self, cols_to_keep):
        return BlockMatrix(self._jbm.filterCols(jarray(Env.jvm().long, cols_to_keep)))

    @handle_py4j
    @typecheck_method(rows_to_keep=listof(integral))
    def filter_rows(self, rows_to_keep):
        return BlockMatrix(self._jbm.filterRows(jarray(Env.jvm().long, rows_to_keep)))
    
    @handle_py4j
    @typecheck_method(rows_to_keep=listof(integral),
                      cols_to_keep=listof(integral))
    def filter(self, rows_to_keep, cols_to_keep):
        return BlockMatrix(self._jbm.filter(jarray(Env.jvm().long, rows_to_keep),
                             jarray(Env.jvm().long, cols_to_keep)))

    @property
    @handle_py4j
    def T(self):
        return BlockMatrix(self._jbm.transpose())

    def cache(self):
        return BlockMatrix(self._jbm.cache())

    @typecheck_method(storage_level=storage_level)
    def persist(self, storage_level='MEMORY_AND_DISK'):
        return BlockMatrix(self._jbm.persist(storage_level))

    def unpersist(self):
        return BlockMatrix(self._jbm.unpersist())

    @handle_py4j
    def to_numpy_matrix(self):
        return numpy_from_breeze(self._jbm.toLocalMatrix())

    @handle_py4j
    @typecheck_method(i=numeric)
    def __mul__(self, i):
        return BlockMatrix(self._jbm.scalarMultiply(float(i)))

    @handle_py4j
    @typecheck_method(i=numeric)
    def __div__(self, i):
        return self * (1./i)

block_matrix_type.set(BlockMatrix)