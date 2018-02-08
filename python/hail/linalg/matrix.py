from hail.utils import new_temp_file, storage_level
from hail.utils.java import Env, handle_py4j, scala_object, jarray, numpy_from_breeze
from hail.typecheck import *
from hail.matrixtable import MatrixTable
from hail.table import Table
from hail.expr.expression import expr_numeric, to_expr, analyze
import numpy as np

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

    @staticmethod
    @handle_py4j
    def _from_numpy_matrix(numpy_matrix, block_size):
        """Create a block matrix from a NumPy matrix.

        Notes
        -----

        This method is useful for creating small block matrices when writing tests. It is not
        efficient for large matrices.

        Parameters
        ----------
        numpy_matrix : NumPy matrix

        Returns
        -------
        :class:`.BlockMatrix`
        """
        # convert to float64 to ensure that jarray method below works
        numpy_matrix = numpy_matrix.astype(np.float64)
        num_rows, num_cols = numpy_matrix.shape[0:2]
        sc = Env.hc()._jsc
        # np.ravel() exports row major by default
        data = jarray(Env.jvm().double, np.ravel(numpy_matrix))
        # breeze is column major
        is_transpose = True

        block_matrix_constructor = getattr(Env.hail().distributedmatrix.BlockMatrix, "from")
        breeze_matrix_constructor = getattr(Env.hail().utils.richUtils.RichDenseMatrixDouble, 'from')

        return BlockMatrix(
            block_matrix_constructor(sc,
                                     breeze_matrix_constructor(num_rows, num_cols, data, is_transpose),
                                     block_size))

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
    @typecheck_method(that=block_matrix_type)
    def __add__(self, that):
        return BlockMatrix(self._jbm.add(that._jbm))

    @handle_py4j
    @typecheck_method(that=block_matrix_type)
    def __sub__(self, that):
        return BlockMatrix(self._jbm.subtract(that._jbm))

    @handle_py4j
    @typecheck_method(that=block_matrix_type)
    def dot(self, that):
        return BlockMatrix(self._jbm.multiply(that._jbm))

    def entries_table(self):
        """Returns a table with the coordinates and numeric value of each block matrix entry.

        Examples
        --------

        >>> from hail.linalg import BlockMatrix
        >>> import numpy as np
        >>> block_matrix = BlockMatrix._from_numpy_matrix(np.matrix([[5, 7], [2, 8]]), 2)
        >>> entries_table = block_matrix.entries_table()
        >>> entries_table.show()
        +--------+--------+-------------+
        |      i |      j |       entry |
        +--------+--------+-------------+
        | !Int64 | !Int64 |    !Float64 |
        +--------+--------+-------------+
        |      0 |      0 | 5.00000e+00 |
        |      0 |      1 | 7.00000e+00 |
        |      1 |      0 | 2.00000e+00 |
        |      1 |      1 | 8.00000e+00 |
        +--------+--------+-------------+

        Warning
        -------
        The resulting table may be filtered, aggregated, and queried, but should only be
        directly exported to disk if the block matrix is very small.

        Returns
        -------
        :class:`.Table`
            Table with a row for each entry.
        """
        hc = Env.hc()
        return Table(self._jbm.entriesTable(hc._jhc))

    @handle_py4j
    @typecheck_method(i=numeric)
    def __mul__(self, i):
        return BlockMatrix(self._jbm.scalarMultiply(float(i)))

    @handle_py4j
    @typecheck_method(i=numeric)
    def __div__(self, i):
        return self * (1. / i)


block_matrix_type.set(BlockMatrix)
