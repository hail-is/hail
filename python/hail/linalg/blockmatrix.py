from hail.utils import new_temp_file, storage_level
from hail.utils.java import Env, scala_object, jarray, numpy_from_breeze, joption, FatalError
from hail.typecheck import *
from hail.matrixtable import MatrixTable
from hail.table import Table
from hail.expr.expression import expr_numeric, to_expr, analyze
import hail.linalg
import numpy as np

block_matrix_type = lazy()


class BlockMatrix(object):
    """Hail's block-distributed matrix of :py:data:`.tfloat64` elements.

    Notes
    -----

    Use ``+`` and ``-`` for block matrix addition and subtraction.

    Use ``*`` for pointwise multiplication of a block matrix and
    a scalar :obj:`int` or :obj:`float`.

    Use :meth:`dot` for block matrix multiplication.

    Blocks are square with side length a common block size.
    Blocks in the final block row or block column may be truncated.
    """

    def __init__(self, jbm):
        self._jbm = jbm

    @classmethod
    def default_block_size(cls):
        """Default block side length."""

        return Env.hail().linalg.BlockMatrix.defaultBlockSize()

    @classmethod
    def read(cls, path):
        """Read a block matrix.

        Parameters
        ----------

        path: :obj:`str`
            Path to input file.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        hc = Env.hc()
        return cls(Env.hail().linalg.BlockMatrix.read(
            hc._jhc, path))

    @staticmethod
    def from_numpy(ndarray, block_size=None):
        """Create a block matrix from a `NumPy ndarray
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`__.

        Warning
        -------

        This method is not efficient for large matrices.

        Parameters
        ----------

        ndarray: NumPy ``ndarray``
            ``ndarray`` with two dimensions, each of non-zero size.
        block_size: :obj:`int`
            Block size, optional.
            Default given by :meth:`default_block_size`.

        Returns
        -------
        :class:`.BlockMatrix`
        """

        if not block_size:
            block_size = BlockMatrix.default_block_size()

        # convert to float64 to ensure that jarray method below works
        ndarray = ndarray.astype(np.float64)
        if ndarray.ndim != 2:
            raise FatalError("from_numpy: ndarray must have two axes, found shape {}".format(ndarray.shape))
        n_rows, n_cols = ndarray.shape[0:2]
        if n_rows==0 or n_cols==0:
            raise FatalError("from_numpy: ndarray dimensions must be non-zero, found shape {}".format(ndarray.shape))

        # np.ravel() exports row major by default
        data = jarray(Env.jvm().double, np.ravel(ndarray))
        # breeze is column major
        is_transpose = True

        sc = Env.hc()._jsc
        block_matrix_constructor = getattr(Env.hail().linalg.BlockMatrix, "from")
        breeze_matrix_constructor = getattr(Env.hail().utils.richUtils.RichDenseMatrixDouble, 'from')

        return BlockMatrix(
            block_matrix_constructor(sc,
                                     breeze_matrix_constructor(n_rows, n_cols, data, is_transpose),
                                     block_size))

    @classmethod
    @typecheck_method(entry_expr=expr_numeric,
                      path=nullable(str),
                      block_size=nullable(int))
    def from_entry_expr(cls, entry_expr, path=None, block_size=None):
        """Create a block matrix from a numeric matrix table entry expression.

        Notes
        -----

        For shuffle resiliency, this functions writes the resulting block matrix
        to disk and then reads the result. By specifying a path, this block
        matrix can be read again using :meth:`BlockMatrix.read`. Otherwise, a
        temporary file is used and then deleted when :func:`hail.stop`
        is called or the program terminates.

        Parameters
        ----------

        entry_expr: :class:`.NumericExpression`
            Numeric entry expression for matrix entries.
        path: :obj:`str`
            Path used to write the resulting block matrix, optional.
            If not specified, a temporary file is used.
        block_size: :obj:`int`
            Block size, optional.
            Default given by :meth:`default_block_size`.

        Returns
        -------
        :class:`.BlockMatrix`
        """

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
        analyze('from_entry_expr', entry_expr, mt._entry_indices)

        mt._jvds.writeBlockMatrix(path, to_expr(entry_expr)._ast.to_hql(), block_size)
        return cls.read(path)

    @staticmethod
    def random(n_rows, n_cols, block_size=None, seed=0, uniform=True):
        """Create a block matrix with uniform or normal random entries.

        Parameters
        ----------

        n_rows: :obj:`int`
            Number of rows.
        n_cols: :obj:`int`
            Number of columns.
        block_size: :obj:`int`
            Block size, optional.
            Default given by :meth:`default_block_size`.
        seed: :obj:`int`
            Random seed.
        uniform: :obj:`bool`
            If ``True``, entries are drawn from the uniform distribution
            on [0,1]. If ``False``, entries are drawn from the standard
            normal distribution.

        Returns
        -------
        :class:`.LocalMatrix`
        """
        if not block_size:
            block_size = BlockMatrix.default_block_size()

        hc = Env.hc()
        return BlockMatrix(Env.hail().linalg.BlockMatrix.random(
            hc._jhc, n_rows, n_cols, block_size, seed, uniform))

    @property
    def n_rows(self):
        """Number of rows.

        Returns
        -------
        :obj:`int`
        """

        return self._jbm.nRows()

    @property
    def n_cols(self):
        """Number of columns.

        Returns
        -------
        :obj:`int`
        """

        return self._jbm.nCols()

    @property
    def block_size(self):
        """Block size.

        Returns
        -------
        :obj:`int`
        """

        return self._jbm.blockSize()

    @typecheck_method(path=str,
                      force_row_major=bool)
    def write(self, path, force_row_major=False):
        """Write the block matrix.

        Parameters
        ----------

        path: :obj:`str`
            Path for output file.
        force_row_major: :obj:`bool`
            If ``True``, transform blocks in column-major format
            to row-major format before writing.
            If ``False``, write blocks in their current format.

        Returns
        -------
        :class:`.LocalMatrix`
        """

        self._jbm.write(path, force_row_major, joption(None))

    @typecheck_method(rows_to_keep=listof(int))
    def filter_rows(self, rows_to_keep):
        """Filter matrix rows.

        Parameters
        ----------

        rows_to_keep: :obj:`list` of :obj:`int`
            Indices of rows to keep. Must be non-empty and increasing.

        Returns
        -------
        :class:`.BlockMatrix`
        """

        return BlockMatrix(self._jbm.filterRows(jarray(Env.jvm().long, rows_to_keep)))

    @typecheck_method(cols_to_keep=listof(int))
    def filter_cols(self, cols_to_keep):
        """Filter matrix columns.

        Parameters
        ----------

        cols_to_keep: :obj:`list` of :obj:`int`
            Indices of columns to keep. Must be non-empty and increasing.

        Returns
        -------
        :class:`.BlockMatrix`
        """

        return BlockMatrix(self._jbm.filterCols(jarray(Env.jvm().long, cols_to_keep)))

    @typecheck_method(rows_to_keep=listof(int),
                      cols_to_keep=listof(int))
    def filter(self, rows_to_keep, cols_to_keep):
        """Filter matrix rows and columns.

        Notes
        -----

        This method has the same effect as :meth:`BlockMatrix.filter_cols`
        followed by :meth:`BlockMatrix.filter_rows` (or vice versa), but
        filters the block matrix in a single pass which may be more efficient.

        Parameters
        ----------
        rows_to_keep: :obj:`list` of :obj:`int`
            Indices of rows to keep. Must be non-empty and increasing.
        cols_to_keep: :obj:`list` of :obj:`int`
            Indices of columns to keep. Must be non-empty and increasing.

        Returns
        -------
        :class:`.BlockMatrix`
        """

        return BlockMatrix(self._jbm.filter(jarray(Env.jvm().long, rows_to_keep),
                                            jarray(Env.jvm().long, cols_to_keep)))


    def to_numpy(self):
        """Create a `NumPy ndarray
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`__
        by collecting the block matrix.

        Warning
        -------

        This method is not efficient for large matrices.

        Returns
        -------
        NumPy ``ndarray``
            ``ndarray`` with shape ``(n_rows, n_cols)``.
        """

        return numpy_from_breeze(self._jbm.toBreezeMatrix())

    @property
    def T(self):
        """Matrix transpose.

        Returns
        -------
        :class:`.LocalMatrix`
        """

        return BlockMatrix(self._jbm.transpose())

    def cache(self):
        """Cache the RDD of blocks.

        Returns
        -------
        :class:`.BlockMatrix`
        """

        return BlockMatrix(self._jbm.cache())

    @typecheck_method(storage_level=storage_level)
    def persist(self, storage_level='MEMORY_AND_DISK'):
        """Persist the RDD of blocks.

        Parameters
        ----------
        storage_level: :str:
          Storage level for persistence.

        Returns
        -------
        :class:`.BlockMatrix`
        """

        return BlockMatrix(self._jbm.persist(storage_level))

    def unpersist(self):
        """Unpersist the RDD of blocks.

        Returns
        -------
        :class:`.BlockMatrix`
        """

        return BlockMatrix(self._jbm.unpersist())

    @typecheck_method(b=block_matrix_type)
    def __add__(self, b):
        """Addition: A + B.

        Parameters
        ----------

        b: :class:`BlockMatrix`

        Returns
        -------
        :class:`.BlockMatrix`
        """

        return BlockMatrix(self._jbm.add(b._jbm))

    @typecheck_method(b=block_matrix_type)
    def __sub__(self, b):
        """Subtraction: A - B.

        Parameters
        ----------

        b: :class:`BlockMatrix`

        Returns
        -------
        :class:`.BlockMatrix`
        """

        return BlockMatrix(self._jbm.subtract(b._jbm))

    @typecheck_method(b=numeric)
    def __mul__(self, b):
        """Scalar multiplication: A * b

        Parameters
        ----------

        b: :obj:`int` or :obj:`float`

        Returns
        -------
        :class:`.BlockMatrix`
        """

        return BlockMatrix(self._jbm.scalarMultiply(float(b)))

    @typecheck_method(b=numeric)
    def __truediv__(self, b):
        """Scalar division: A / b

        Parameters
        ----------

        b: :obj:`int` or :obj:`float`

        Returns
        -------
        :class:`.BlockMatrix`
        """

        return self * (1. / b)

    @typecheck_method(b=block_matrix_type)
    def dot(self, b):
        """Matrix multiplication: A.dot(B).

        Parameters
        ----------

        b: :class:`BlockMatrix`

        Returns
        -------
        :class:`.BlockMatrix`
        """

        return BlockMatrix(self._jbm.multiply(b._jbm))

    def entries(self):
        """Returns a table with the coordinates and numeric value of each block matrix entry.

        Examples
        --------

        >>> from hail.linalg import BlockMatrix
        >>> import numpy as np
        >>> block_matrix = BlockMatrix.from_numpy(np.matrix([[5, 7], [2, 8]]), 2)
        >>> entries_table = block_matrix.entries()
        >>> entries_table.show()
        +-------+-------+-------------+
        |     i |     j |       entry |
        +-------+-------+-------------+
        | int64 | int64 |     float64 |
        +-------+-------+-------------+
        |     0 |     0 | 5.00000e+00 |
        |     0 |     1 | 7.00000e+00 |
        |     1 |     0 | 2.00000e+00 |
        |     1 |     1 | 8.00000e+00 |
        +-------+-------+-------------+

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

block_matrix_type.set(BlockMatrix)
