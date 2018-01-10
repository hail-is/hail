from hail.utils.java import Env, handle_py4j, scala_object, jarray
from hail.typecheck import *
from hail.api2 import MatrixTable
from hail.expr.expression import expr_numeric, to_expr, analyze

block_matrix_type = lazy()

class BlockMatrix(object):

    @staticmethod
    @handle_py4j
    def default_block_size():
        return scala_object(Env.hail().distributedmatrix, "BlockMatrix").defaultBlockSize()

    @staticmethod
    @handle_py4j
    def read(path):
        hc = Env.hc()
        return BlockMatrix(Env.hail().distributedmatrix.BlockMatrix.read(
            hc._jhc, path))


    @staticmethod
    @handle_py4j
    def random(num_rows, num_cols, block_size):
        hc = Env.hc()
        return BlockMatrix(scala_object(Env.hail().distributedmatrix, 'BlockMatrix').random(
                hc._jhc, num_rows, num_cols, block_size))

    def __init__(self, jbm):
        self._jbm = jbm

    @property
    @handle_py4j
    def num_rows(self):
        return self._jbm.rows()

    @property
    @handle_py4j
    def num_cols(self):
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

    @handle_py4j
    def transpose(self):
        return BlockMatrix(self._jbm.transpose())

    def cache(self):
        return BlockMatrix(self._jbm.cache())

    @typecheck_method(storage_level=enumeration('NONE', 'DISK_ONLY', 'DISK_ONLY_2', 'MEMORY_ONLY',
                                                'MEMORY_ONLY_2', 'MEMORY_ONLY_SER', 'MEMORY_ONLY_SER_2',
                                                'MEMORY_AND_DISK', 'MEMORY_AND_DISK_2', 'MEMORY_AND_DISK_SER',
                                                'MEMORY_AND_DISK_SER_2', 'OFF_HEAP'))
    def persist(self, storage_level='MEMORY_AND_DISK'):
        return BlockMatrix(self._jbm.persist(storage_level))

    def unpersist(self):
        return BlockMatrix(self._jbm.unpersist())

block_matrix_type.set(BlockMatrix)


@staticmethod
@handle_py4j
@typecheck(entry_expr=expr_numeric,
           path=strlike,
           block_size=nullable(integral))
def block_matrix_from_expr(entry_expr,
                           path,
                           block_size=None):
    if not block_size:
        block_size = BlockMatrix.default_block_size()
    source = entry_expr._indices.source
    if not isinstance(source, MatrixTable):
        raise ValueError("Expect an expression of 'MatrixTable', found {}".format(
            "expression of '{}'".format(source.__class__) if source is not None else 'scalar expression'))
    mt = source
    base, _ = mt._process_joins(entry_expr)
    analyze(entry_expr, mt._entry_indices)

    mt._jvds.writeBlockMatrix(path, to_expr(entry_expr)._ast.to_hql(), block_size)
    return BlockMatrix(BlockMatrix.read(path))
