from hail.utils import new_temp_file, new_local_temp_file, local_path_uri, storage_level
from hail.utils.java import Env, jarray, joption
from hail.typecheck import *
from hail.table import Table
from hail.expr.expressions import expr_float64, matrix_table_source, check_entry_indexed
import numpy as np
from enum import IntEnum

block_matrix_type = lazy()


class Form(IntEnum):
    SCALAR = 0
    COLUMN = 1
    ROW = 2
    MATRIX = 3

    @classmethod
    def of(cls, shape):
        assert len(shape) == 2
        if shape[0] == 1 and shape[1] == 1:
            return Form.SCALAR
        elif shape[1] == 1:
            return Form.COLUMN
        elif shape[0] == 1:
            return Form.ROW
        else:
            return Form.MATRIX

    @staticmethod
    def compatible(shape_a, shape_b, op):
        form_a = Form.of(shape_a)
        form_b = Form.of(shape_b)
        if (form_a == Form.SCALAR or
                form_b == Form.SCALAR or
                form_a == form_b and shape_a == shape_b or
                {form_a, form_b} == {Form.MATRIX, Form.COLUMN} and shape_a[0] == shape_b[0] or
                {form_a, form_b} == {Form.MATRIX, Form.ROW} and shape_a[1] == shape_b[1]):
            return form_a, form_b
        else:
            raise ValueError(f'incompatible shapes for {op}: {shape_a} and {shape_b}')


class BlockMatrix(object):
    """Hail's block-distributed matrix of :py:data:`.tfloat64` elements.

    .. include:: ../_templates/experimental.rst

    Notes
    -----
    A block matrix is a distributed analogue of a two-dimensional
    `NumPy ndarray
    <https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html>`__ with
    shape ``(n_rows, n_cols)`` and dtype ``float64``. The core operations are
    consistent with NumPy: ``+``, ``-``, ``*``, and ``/`` for element-wise
    addition, subtraction, multiplication, and division; ``@`` for matrix
    multiplication; and ``**`` for element-wise exponentiation to a scalar
    power.

    For element-wise binary operations, each operand may be a block matrix, an
    ndarray, or a scalar (:obj:`int` or :obj:`float`). For matrix
    multiplication, each operand may be a block matrix or an ndarray. If either
    operand is a block matrix, the result is a block matrix.

    To interoperate with block matrices, ndarray operands must be one or two
    dimensional with dtype convertible to ``float64``. One-dimensional ndarrays
    of shape ``(n)`` are promoted to two-dimensional ndarrays of shape ``(1,
    n)``, i.e. a single row.

    Block matrices support broadcasting of ``+``, ``-``, ``*``, and ``/``
    between matrices of different shapes, consistent with the NumPy
    `broadcasting rules
    <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`__.
    There is one exception: block matrices do not currently support element-wise
    "outer product" of a single row and a single column, although the same
    effect can be achieved for ``*`` by using ``@``.

    Under the hood, block matrices are partitioned like a checkerboard into
    square blocks with side length a common block size (blocks in the final row
    or column of blocks may be truncated). Block size defaults to the value
    given by :meth:`default_block_size`. Binary operations between block
    matrices require that both operands have the same block size.

    Warning
    -------
    For binary operations, if the first operand is an ndarray and the second
    operand is a block matrix, the result will be a ndarray of block matrices.
    To achieve the desired behavior for ``+`` and ``*``, place the block matrix
    operand first; for ``-``, ``/``, and ``@``, first convert the ndarray to a
    block matrix using :meth:`.from_numpy`.
    """

    def __init__(self, jbm):
        self._jbm = jbm

    @classmethod
    @typecheck_method(path=str)
    def read(cls, path):
        """Reads a block matrix.

        Parameters
        ----------
        path: :obj:`str`
            Path to input file.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return cls(Env.hail().linalg.BlockMatrix.read(Env.hc()._jhc, path))

    @classmethod
    @typecheck_method(uri=str,
                      n_rows=int,
                      n_cols=int,
                      block_size=nullable(int))
    def fromfile(cls, uri, n_rows, n_cols, block_size=None):
        """Creates a block matrix from a binary file.

        Examples
        --------

        >>> import numpy as np
        >>> a = np.random.rand(10, 20)
        >>> a.tofile('/local/file') # doctest: +SKIP

        To create a block matrix of the same dimensions:

        >>> from hail.linalg import BlockMatrix
        >>> bm = BlockMatrix.fromfile('file:///local/file', 10, 20) # doctest: +SKIP

        Notes
        -----
        This method, analogous to `numpy.fromfile
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.fromfile.html>`__,
        reads a binary file of float64 values in row-major order, such as that
        produced by `numpy.tofile
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tofile.html>`__
        or :meth:`BlockMatrix.tofile`.

        Binary files produced and consumed by :meth:`.tofile` and
        :meth:`.fromfile` are not platform independent, so should only be used
        for inter-operating with NumPy, not storage. Use
        :meth:`BlockMatrix.write` and :meth:`BlockMatrix.read` to save and load
        block matrices, since these methods write and read blocks in parallel
        and are platform independent.

        A NumPy ndarray must have type float64 for the output of
        func:`numpy.tofile` to be a valid binary input to :meth:`.fromfile`.
        This is not checked.

        The number of entries must be less than :math:`2^{31}`.

        Parameters
        ----------
        uri: :obj:`str`, optional
            URI of binary input file.
        n_rows: :obj:`int`
            Number of rows.
        n_cols: :obj:`int`
            Number of columns.
        block_size: :obj:`int`, optional
            Block size. Default given by :meth:`default_block_size`.

        See Also
        --------
        :meth:`.from_numpy`
        """
        if not block_size:
            block_size = BlockMatrix.default_block_size()

        n_entries = n_rows * n_cols
        if n_entries >= 1 << 31:
            raise ValueError(f'number of entries must be less than 2^31, found {n_entries}')

        hc = Env.hc()
        bdm = Env.hail().utils.richUtils.RichDenseMatrixDouble.importFromDoubles(hc._jhc, uri, n_rows, n_cols, True)
        return cls(Env.hail().linalg.BlockMatrix.fromBreezeMatrix(hc._jsc, bdm, block_size))

    @classmethod
    @typecheck_method(ndarray=np.ndarray,
                      block_size=nullable(int))
    def from_numpy(cls, ndarray, block_size=None):
        """Distributes a `NumPy ndarray
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`__
        as a block matrix.

        Examples
        --------

        >>> import numpy as np
        >>> a = np.random.rand(10, 20)
        >>> bm = BlockMatrix.from_numpy(a)

        Notes
        -----
        The ndarray must have two dimensions, each of non-zero size.

        The number of entries must be less than :math:`2^{31}`.

        Parameters
        ----------
        ndarray: :class:`numpy.ndarray`
            ndarray with two dimensions, each of non-zero size.
        block_size: :obj:`int`, optional
            Block size. Default given by :meth:`default_block_size`.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        if not block_size:
            block_size = BlockMatrix.default_block_size()

        if any(i == 0 for i in ndarray.shape):
            raise ValueError(f'from_numpy: ndarray dimensions must be non-zero, found shape {ndarray.shape}')

        nd = _ndarray_as_2d(ndarray)
        nd = _ndarray_as_float64(nd)
        n_rows, n_cols = nd.shape

        path = new_local_temp_file()
        uri = local_path_uri(path)
        nd.tofile(path)
        return cls.fromfile(uri, n_rows, n_cols, block_size)

    @classmethod
    @typecheck_method(entry_expr=expr_float64,
                      block_size=nullable(int))
    def from_entry_expr(cls, entry_expr, block_size=None):
        """Create a block matrix using a matrix table entry expression.

        Parameters
        ----------
        entry_expr: :class:`.Float64Expression`
            Entry expression for numeric matrix entries.
        block_size: :obj:`int`, optional
            Block size. Default given by :meth:`.BlockMatrix.default_block_size`.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        path = new_temp_file()
        cls.write_from_entry_expr(entry_expr, path, block_size)
        return cls.read(path)

    @classmethod
    @typecheck_method(n_rows=int,
                      n_cols=int,
                      block_size=nullable(int),
                      seed=int,
                      uniform=bool)
    def random(cls, n_rows, n_cols, block_size=None, seed=0, uniform=True):
        """Creates a block matrix with uniform or normal random entries.

        Parameters
        ----------
        n_rows: :obj:`int`
            Number of rows.
        n_cols: :obj:`int`
            Number of columns.
        block_size: :obj:`int`, optional
            Block size. Default given by :meth:`default_block_size`.
        seed: :obj:`int`
            Random seed.
        uniform: :obj:`bool`
            If ``True``, entries are drawn from the uniform distribution
            on [0,1]. If ``False``, entries are drawn from the standard
            normal distribution.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        if not block_size:
            block_size = BlockMatrix.default_block_size()
        return cls(Env.hail().linalg.BlockMatrix.random(Env.hc()._jhc, n_rows, n_cols, block_size, seed, uniform))

    @classmethod
    @typecheck_method(n_rows=int,
                      n_cols=int,
                      data=sequenceof(float),
                      row_major=bool,
                      block_size=int)
    def _create(cls, n_rows, n_cols, data, row_major, block_size):
        """Private method for creating small test matrices."""

        bdm = Env.hail().utils.richUtils.RichDenseMatrixDouble.apply(n_rows,
                                                                     n_cols,
                                                                     jarray(Env.jvm().double, data),
                                                                     row_major)
        return cls(Env.hail().linalg.BlockMatrix.fromBreezeMatrix(Env.hc()._jsc, bdm, block_size))

    @staticmethod
    def default_block_size():
        """Default block side length."""
        return Env.hail().linalg.BlockMatrix.defaultBlockSize()

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
    def shape(self):
        """Shape of matrix.

        Returns
        -------
        (:obj:`int`, :obj:`int`)
           Number of rows and number of columns.
        """
        return self.n_rows, self.n_cols

    @property
    def block_size(self):
        """Block size.

        Returns
        -------
        :obj:`int`
        """
        return self._jbm.blockSize()

    @property
    def _jdata(self):
        return self._jbm.toBreezeMatrix().data()

    @property
    def _as_scalar(self):
        assert self.n_rows == 1 and self.n_cols == 1
        return self._jbm.toBreezeMatrix().apply(0, 0)

    @typecheck_method(path=str,
                      force_row_major=bool)
    def write(self, path, force_row_major=False):
        """Writes the block matrix.

        Parameters
        ----------
        path: :obj:`str`
            Path for output file.
        force_row_major: :obj:`bool`
            If ``True``, transform blocks in column-major format
            to row-major format before writing.
            If ``False``, write blocks in their current format.
        """
        self._jbm.write(path, force_row_major, joption(None))

    @staticmethod
    @typecheck(entry_expr=expr_float64,
               path=str,
               block_size=nullable(int))
    def write_from_entry_expr(entry_expr, path, block_size=None):
        """Writes a block matrix from a matrix table entry expression.

        Notes
        -----
        The resulting file can be loaded with :meth:`BlockMatrix.read`.

        Parameters
        ----------
        entry_expr: :class:`.Float64Expression`
            Entry expression for numeric matrix entries.
        path: :obj:`str`
            Path for output.
        block_size: :obj:`int`, optional
            Block size. Default given by :meth:`.BlockMatrix.default_block_size`.
        """
        if not block_size:
            block_size = BlockMatrix.default_block_size()

        check_entry_indexed('write_from_entry_expr/entry_expr', entry_expr)

        mt = matrix_table_source('write_from_entry_expr/entry_expr', entry_expr)

        #  FIXME: remove once select_entries on a field is free
        if entry_expr in mt._fields_inverse:
            field = mt._fields_inverse[entry_expr]
            mt._jvds.writeBlockMatrix(path, field, block_size)
        else:
            field = Env.get_uid()
            mt.select_entries(**{field: entry_expr})._jvds.writeBlockMatrix(path, field, block_size)

    @typecheck_method(rows_to_keep=sequenceof(int))
    def filter_rows(self, rows_to_keep):
        """Filters matrix rows.

        Parameters
        ----------
        rows_to_keep: :obj:`list` of :obj:`int`
            Indices of rows to keep. Must be non-empty and increasing.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return BlockMatrix(self._jbm.filterRows(jarray(Env.jvm().long, rows_to_keep)))

    @typecheck_method(cols_to_keep=sequenceof(int))
    def filter_cols(self, cols_to_keep):
        """Filters matrix columns.

        Parameters
        ----------
        cols_to_keep: :obj:`list` of :obj:`int`
            Indices of columns to keep. Must be non-empty and increasing.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return BlockMatrix(self._jbm.filterCols(jarray(Env.jvm().long, cols_to_keep)))

    @typecheck_method(rows_to_keep=sequenceof(int),
                      cols_to_keep=sequenceof(int))
    def filter(self, rows_to_keep, cols_to_keep):
        """Filters matrix rows and columns.

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

    @typecheck_method(table=Table,
                      radius=int,
                      include_diagonal=bool)
    def _filtered_entries_table(self, table, radius, include_diagonal):
        return Table(self._jbm.filteredEntriesTable(table._jt, radius, include_diagonal))

    @typecheck_method(uri=str)
    def tofile(self, uri):
        """Collects and writes data to a binary file.

        Examples
        --------

        >>> from hail.linalg import BlockMatrix
        >>> import numpy as np
        >>> bm = BlockMatrix.random(10, 20)
        >>> bm.tofile('file:///local/file') # doctest: +SKIP

        To create a :class:`numpy.ndarray` of the same dimensions:

        >>> a = np.fromfile('/local/file').reshape((10, 20)) # doctest: +SKIP

        Notes
        -----
        This method, analogous to `numpy.tofile
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tofile.html>`__,
        produces a binary file of float64 values in row-major order, which can
        be read by functions such as `numpy.fromfile
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.fromfile.html>`__
        (if a local file) and :meth:`BlockMatrix.fromfile`.

        Binary files produced and consumed by :meth:`.tofile` and
        :meth:`.fromfile` are not platform independent, so should only be used
        for inter-operating with NumPy, not storage. Use
        :meth:`BlockMatrix.write` and :meth:`BlockMatrix.read` to save and load
        block matrices, since these methods write and read blocks in parallel
        and are platform independent.

        The number of entries must be less than :math:`2^{31}`.

        Parameters
        ----------
        uri: :obj:`str`, optional
            URI of binary output file.

        See Also
        --------
        :meth:`.to_numpy`
        """
        n_entries = self.n_rows * self.n_cols
        if n_entries >= 2 << 31:
            raise ValueError(f'number of entries must be less than 2^31, found {n_entries}')

        bdm = self._jbm.toBreezeMatrix()
        row_major = Env.hail().utils.richUtils.RichDenseMatrixDouble.exportToDoubles(Env.hc()._jhc, uri, bdm, True)
        assert row_major

    def to_numpy(self):
        """Collects the block matrix into a `NumPy ndarray
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`__.

        Examples
        --------

        >>> from hail.linalg import BlockMatrix
        >>> bm = BlockMatrix.random(10, 20)
        >>> a = bm.to_numpy()

        Notes
        -----
        The number of entries must be less than :math:`2^{31}`.

        The resulting ndarray will have the same shape as the block matrix.

        Returns
        -------
        :class:`numpy.ndarray`
        """
        path = new_local_temp_file()
        uri = local_path_uri(path)
        self.tofile(uri)
        return np.fromfile(path).reshape((self.n_rows, self.n_cols))

    @property
    def T(self):
        """Matrix transpose.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return BlockMatrix(self._jbm.transpose())

    def cache(self):
        """Persist this block matrix in memory.

        Notes
        -----
        This method is an alias for :meth:`persist("MEMORY_ONLY") <hail.linalg.BlockMatrix.persist>`.

        Returns
        -------
        :class:`.BlockMatrix`
            Cached block matrix.
        """
        return self.persist('MEMORY_ONLY')

    @typecheck_method(storage_level=storage_level)
    def persist(self, storage_level='MEMORY_AND_DISK'):
        """Persists this block matrix in memory or on disk.

        Notes
        -----
        The :meth:`.BlockMatrix.persist` and :meth:`.BlockMatrix.cache`
        methods store the current block matrix on disk or in memory temporarily
        to avoid redundant computation and improve the performance of Hail
        pipelines. This method is not a substitution for
        :meth:`.BlockMatrix.write`, which stores a permanent file.

        Most users should use the "MEMORY_AND_DISK" storage level. See the `Spark
        documentation
        <http://spark.apache.org/docs/latest/programming-guide.html#rdd-persistence>`__
        for a more in-depth discussion of persisting data.

        Parameters
        ----------
        storage_level : str
            Storage level.  One of: NONE, DISK_ONLY,
            DISK_ONLY_2, MEMORY_ONLY, MEMORY_ONLY_2, MEMORY_ONLY_SER,
            MEMORY_ONLY_SER_2, MEMORY_AND_DISK, MEMORY_AND_DISK_2,
            MEMORY_AND_DISK_SER, MEMORY_AND_DISK_SER_2, OFF_HEAP

        Returns
        -------
        :class:`.BlockMatrix`
            Persisted block matrix.
        """
        return BlockMatrix(self._jbm.persist(storage_level))

    def unpersist(self):
        """Unpersists this block matrix from memory/disk.

        Notes
        -----
        This function will have no effect on a dataset that was not previously
        persisted.

        Returns
        -------
        :class:`.BlockMatrix`
            Unpersisted block matrix.
        """
        return BlockMatrix(self._jbm.unpersist())

    def __pos__(self):
        return self

    def __neg__(self):
        """Negation: -a.

        Returns
        -------
        :class:`.BlockMatrix`
        """

        op = getattr(self._jbm, "unary_$minus")
        return BlockMatrix(op())

    def _promote(self, b, op, reverse=False):
        a = self
        form_a, form_b = Form.compatible(a.shape, _shape(b), op)

        if form_b > form_a:
            if isinstance(b, np.ndarray):
                b = BlockMatrix.from_numpy(b, a.block_size)
            return b._promote(a, op, reverse=True)

        assert form_a >= form_b

        if form_b == Form.SCALAR:
            if isinstance(b, int) or isinstance(b, float):
                b = float(b)
            elif isinstance(b, np.ndarray):
                b = _ndarray_as_float64(b).item()
            else:
                b = b._as_scalar
        elif form_a > form_b:
            if isinstance(b, np.ndarray):
                b = _jarray_from_ndarray(b)
            else:
                assert isinstance(b, BlockMatrix)
                b = b._jdata
        else:
            assert form_a == form_b
            if not isinstance(b, BlockMatrix):
                assert isinstance(b, np.ndarray)
                b = BlockMatrix.from_numpy(b, a.block_size)

        assert (isinstance(a, BlockMatrix) and 
                (isinstance(b, BlockMatrix) or isinstance(b, float) or b.getClass().isArray()) and
                (not (isinstance(b, BlockMatrix) and reverse)))

        return a, b, form_b, reverse

    @typecheck_method(b=oneof(numeric, np.ndarray, block_matrix_type))
    def __add__(self, b):
        """Addition: a + b.

        Parameters
        ----------
        b: :obj:`int` or :obj:`float` or :class:`numpy.ndarray` or :class:`BlockMatrix`

        Returns
        -------
        :class:`.BlockMatrix`
        """
        new_a, new_b, form_b, _ = self._promote(b, 'addition')

        if isinstance(new_b, float):
            return BlockMatrix(new_a._jbm.scalarAdd(new_b))
        elif isinstance(new_b, BlockMatrix):
            return BlockMatrix(new_a._jbm.add(new_b._jbm))
        else:
            assert new_b.getClass().isArray()
            if form_b == Form.COLUMN:
                return BlockMatrix(new_a._jbm.colVectorAdd(new_b))
            else:
                assert form_b == Form.ROW
                return BlockMatrix(new_a._jbm.rowVectorAdd(new_b))

    @typecheck_method(b=oneof(numeric, np.ndarray, block_matrix_type))
    def __sub__(self, b):
        """Subtraction: a - b.

        Parameters
        ----------
        b: :obj:`int` or :obj:`float` or :class:`numpy.ndarray` or :class:`BlockMatrix`

        Returns
        -------
        :class:`.BlockMatrix`
        """
        new_a, new_b, form_b, reverse = self._promote(b, 'subtraction')

        if isinstance(new_b, float):
            if reverse:
                return BlockMatrix(new_a._jbm.reverseScalarSub(new_b))
            else:
                return BlockMatrix(new_a._jbm.scalarSub(new_b))
        elif isinstance(new_b, BlockMatrix):
            assert not reverse
            return BlockMatrix(new_a._jbm.sub(new_b._jbm))
        else:
            assert new_b.getClass().isArray()
            if form_b == Form.COLUMN:
                if reverse:
                    return BlockMatrix(new_a._jbm.reverseColVectorSub(new_b))
                else:
                    return BlockMatrix(new_a._jbm.colVectorSub(new_b))
            else:
                assert form_b == Form.ROW
                if reverse:
                    return BlockMatrix(new_a._jbm.reverseRowVectorSub(new_b))
                else:
                    return BlockMatrix(new_a._jbm.rowVectorSub(new_b))

    @typecheck_method(b=oneof(numeric, np.ndarray, block_matrix_type))
    def __mul__(self, b):
        """Element-wise multiplication: a * b.

        Parameters
        ----------
        b: :obj:`int` or :obj:`float` or :class:`numpy.ndarray` or :class:`BlockMatrix`

        Returns
        -------
        :class:`.BlockMatrix`
        """
        new_a, new_b, form_b, _ = self._promote(b, 'element-wise multiplication')

        if isinstance(new_b, float):
            return BlockMatrix(new_a._jbm.scalarMul(new_b))
        elif isinstance(new_b, BlockMatrix):
            return BlockMatrix(new_a._jbm.mul(new_b._jbm))
        else:
            assert new_b.getClass().isArray()
            if form_b == Form.COLUMN:
                return BlockMatrix(new_a._jbm.colVectorMul(new_b))
            else:
                assert form_b == Form.ROW
                return BlockMatrix(new_a._jbm.rowVectorMul(new_b))

    @typecheck_method(b=oneof(numeric, np.ndarray, block_matrix_type))
    def __truediv__(self, b):
        """Element-wise division: a / b.

        Parameters
        ----------
        b: :obj:`int` or :obj:`float` or :class:`numpy.ndarray` or :class:`BlockMatrix`

        Returns
        -------
        :class:`.BlockMatrix`
        """
        new_a, new_b, form_b, reverse = self._promote(b, 'element-wise division')

        if isinstance(new_b, float):
            if reverse:
                return BlockMatrix(new_a._jbm.reverseScalarDiv(new_b))
            else:
                return BlockMatrix(new_a._jbm.scalarDiv(new_b))
        elif isinstance(new_b, BlockMatrix):
            assert not reverse
            return BlockMatrix(new_a._jbm.div(new_b._jbm))
        else:
            assert new_b.getClass().isArray()
            if form_b == Form.COLUMN:
                if reverse:
                    return BlockMatrix(new_a._jbm.reverseColVectorDiv(new_b))
                else:
                    return BlockMatrix(new_a._jbm.colVectorDiv(new_b))
            else:
                assert form_b == Form.ROW
                if reverse:
                    return BlockMatrix(new_a._jbm.reverseRowVectorDiv(new_b))
                else:
                    return BlockMatrix(new_a._jbm.rowVectorDiv(new_b))

    @typecheck_method(b=numeric)
    def __radd__(self, b):
        return self + b

    @typecheck_method(b=numeric)
    def __rsub__(self, b):
        return BlockMatrix(self._jbm.reverseScalarSub(float(b)))

    @typecheck_method(b=numeric)
    def __rmul__(self, b):
        return self * b

    @typecheck_method(b=numeric)
    def __rtruediv__(self, b):
        return BlockMatrix(self._jbm.reverseScalarDiv(float(b)))

    @typecheck_method(b=oneof(np.ndarray, block_matrix_type))
    def __matmul__(self, b):
        """Matrix multiplication: a @ b.

        Parameters
        ----------
        b: :class:`numpy.ndarray` or :class:`BlockMatrix`

        Returns
        -------
        :class:`.BlockMatrix`
        """
        if isinstance(b, np.ndarray):
            return self @ BlockMatrix.from_numpy(b, self.block_size)
        else:
            if self.n_cols != b.n_rows:
                raise ValueError(f'incompatible shapes for matrix multiplication: {self.shape} and {b.shape}')
            return BlockMatrix(self._jbm.dot(b._jbm))

    @typecheck_method(x=numeric)
    def __pow__(self, x):
        """Element-wise exponentiation: a ** x.

        Parameters
        ----------
        x: :obj:`int` or :obj:`float`
            Exponent.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return BlockMatrix(self._jbm.pow(float(x)))

    def sqrt(self):
        """Element-wise square root.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return BlockMatrix(self._jbm.sqrt())

    def diagonal(self):
        """Extracts diagonal elements as ndarray.

        Returns
        -------
        :class:`numpy.ndarray`
        """
        return _ndarray_from_jarray(self._jbm.diagonal())

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
        return Table(self._jbm.entriesTable(Env.hc()._jhc))


block_matrix_type.set(BlockMatrix)


def _shape(b):
    if isinstance(b, int) or isinstance(b, float):
        return 1, 1
    if isinstance(b, np.ndarray):
        b = _ndarray_as_2d(b)
    else:
        isinstance(b, BlockMatrix)
    return b.shape


def _ndarray_as_2d(nd):
    if nd.ndim == 1:
        nd = nd.reshape(1, nd.shape[0])
    elif nd.ndim > 2:
        raise ValueError(f'ndarray must have one or two axes, found shape {nd.shape}')
    return nd


def _ndarray_as_float64(nd):
    if nd.dtype != np.float64:
        try:
            nd = nd.astype(np.float64)
        except ValueError as e:
            raise TypeError(f"ndarray elements of dtype {nd.dtype} cannot be converted to type 'float64'") from e
    return nd


def _jarray_from_ndarray(nd):
    if nd.size >= (1 << 31):
        raise ValueError(f'size of ndarray must be less than 2^31, found {nd.size}')

    nd = _ndarray_as_float64(nd)
    path = new_local_temp_file()
    uri = local_path_uri(path)
    nd.tofile(path)
    return Env.hail().utils.richUtils.RichArray.importFromDoubles(Env.hc()._jhc, uri, nd.size)


def _ndarray_from_jarray(ja):
    path = new_local_temp_file()
    uri = local_path_uri(path)
    Env.hail().utils.richUtils.RichArray.exportToDoubles(Env.hc()._jhc, uri, ja)
    return np.fromfile(path)
