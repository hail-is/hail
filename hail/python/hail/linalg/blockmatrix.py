import os

import itertools
import math
import re
import numpy as np
import scipy.linalg as spla

import hail as hl
import hail.expr.aggregators as agg
from hail.expr import construct_expr, construct_variable
from hail.expr.blockmatrix_type import tblockmatrix
from hail.expr.expressions import (expr_float64, matrix_table_source, expr_ndarray,
                                   check_entry_indexed, expr_tuple, expr_array, expr_int32, expr_int64)
from hail.ir import (BlockMatrixWrite, BlockMatrixMap2, ApplyBinaryPrimOp, F64,
                     BlockMatrixBroadcast, ValueToBlockMatrix, BlockMatrixRead,
                     BlockMatrixMap, ApplyUnaryPrimOp, BlockMatrixDot, BlockMatrixCollect,
                     tensor_shape_to_matrix_shape, BlockMatrixAgg, BlockMatrixRandom,
                     BlockMatrixToValueApply, BlockMatrixToTable, BlockMatrixFilter,
                     TableFromBlockMatrixNativeReader, TableRead, BlockMatrixSlice,
                     BlockMatrixSparsify, BlockMatrixDensify, RectangleSparsifier,
                     RowIntervalSparsifier, BandSparsifier, PerBlockSparsifier)
from hail.ir.blockmatrix_reader import BlockMatrixNativeReader, BlockMatrixBinaryReader
from hail.ir.blockmatrix_writer import (BlockMatrixBinaryWriter,
                                        BlockMatrixNativeWriter, BlockMatrixRectanglesWriter)
from hail.ir import ExportType
from hail.table import Table
from hail.typecheck import (typecheck, typecheck_method, nullable, oneof,
                            sliceof, sequenceof, lazy, enumeration, numeric, tupleof, func_spec,
                            sized_tupleof)
from hail.utils import (new_temp_file, local_path_uri, storage_level, with_local_temp_file,
                        new_local_temp_file)
from hail.utils.java import Env

block_matrix_type = lazy()


class BlockMatrix(object):
    """Hail's block-distributed matrix of :py:data:`.tfloat64` elements.

    .. include:: ../_templates/experimental.rst

    A block matrix is a distributed analogue of a two-dimensional
    `NumPy ndarray
    <https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html>`__ with
    shape ``(n_rows, n_cols)`` and NumPy dtype ``float64``.
    Import the class with:

    >>> from hail.linalg import BlockMatrix

    Under the hood, block matrices are partitioned like a checkerboard into
    square blocks with side length a common block size. Blocks in the final row
    or column of blocks may be truncated, so block size need not evenly divide
    the matrix dimensions. Block size defaults to the value given by
    :meth:`default_block_size`.

    **Operations and broadcasting**

    The core operations are consistent with NumPy: ``+``, ``-``, ``*``, and
    ``/`` for element-wise addition, subtraction, multiplication, and division;
    ``@`` for matrix multiplication; ``T`` for transpose; and ``**`` for
    element-wise exponentiation to a scalar power.

    For element-wise binary operations, each operand may be a block matrix, an
    ndarray, or a scalar (:obj:`int` or :obj:`float`). For matrix
    multiplication, each operand may be a block matrix or an ndarray. If either
    operand is a block matrix, the result is a block matrix. Binary operations
    between block matrices require that both operands have the same block size.

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

    Warning
    -------

        For binary operations, if the first operand is an ndarray and the
        second operand is a block matrix, the result will be a ndarray of block
        matrices. To achieve the desired behavior for ``+`` and ``*``, place the
        block matrix operand first; for ``-``, ``/``, and ``@``, first convert
        the ndarray to a block matrix using :meth:`.from_numpy`.

    Warning
    -------

        Block matrix multiplication requires special care due to each block
        of each operand being a dependency of multiple blocks in the product.

        The :math:`(i, j)`-block in the product ``a @ b`` is computed by summing
        the products of corresponding blocks in block row :math:`i` of ``a`` and
        block column :math:`j` of ``b``. So overall, in addition to this
        multiplication and addition, the evaluation of ``a @ b`` realizes each
        block of ``a`` as many times as the number of block columns of ``b``
        and realizes each block of ``b`` as many times as the number of
        block rows of ``a``.

        This becomes a performance and resilience issue whenever ``a`` or ``b``
        is defined in terms of pending transformations (such as linear
        algebra operations). For example, evaluating ``a @ (c @ d)`` will
        effectively evaluate ``c @ d`` as many times as the number of block rows
        in ``a``.

        To limit re-computation, write or cache transformed block matrix
        operands before feeding them into matrix multiplication:

        >>> c = BlockMatrix.read('c.bm')      # doctest: +SKIP
        >>> d = BlockMatrix.read('d.bm')      # doctest: +SKIP
        >>> (c @ d).write('cd.bm')            # doctest: +SKIP
        >>> a = BlockMatrix.read('a.bm')      # doctest: +SKIP
        >>> e = a @ BlockMatrix.read('cd.bm') # doctest: +SKIP

    **Indexing and slicing**

    Block matrices also support NumPy-style 2-dimensional
    `indexing and slicing <https://docs.scipy.org/doc/numpy/user/basics.indexing.html>`__,
    with two differences.
    First, slices ``start:stop:step`` must be non-empty with positive ``step``.
    Second, even if only one index is a slice, the resulting block matrix is still
    2-dimensional.

    For example, for a block matrix ``bm`` with 10 rows and 10 columns:

     - ``bm[0, 0]`` is the element in row 0 and column 0 of ``bm``.

     - ``bm[0:1, 0]`` is a block matrix with 1 row, 1 column,
       and element ``bm[0, 0]``.

     - ``bm[2, :]`` is a block matrix with 1 row, 10 columns,
       and elements from row 2 of ``bm``.

     - ``bm[:3, -1]`` is a block matrix with 3 rows, 1 column,
       and the first 3 elements of the last column of ``bm``.

     - ``bm[::2, ::2]`` is a block matrix with 5 rows, 5 columns,
       and all evenly-indexed elements of ``bm``.

    Use :meth:`filter`, :meth:`filter_rows`, and :meth:`filter_cols` to
    subset to non-slice subsets of rows and columns, e.g. to rows ``[0, 2, 5]``.

    **Block-sparse representation**

    By default, block matrices compute and store all blocks explicitly.
    However, some applications involve block matrices in which:

    - some blocks consist entirely of zeroes.

    - some blocks are not of interest.

    For example, statistical geneticists often want to compute and manipulate a
    banded correlation matrix capturing "linkage disequilibrium" between nearby
    variants along the genome. In this case, working with the full correlation
    matrix for tens of millions of variants would be prohibitively expensive,
    and in any case, entries far from the diagonal are either not of interest or
    ought to be zeroed out before downstream linear algebra.

    To enable such computations, block matrices do not require that all blocks
    be realized explicitly. Implicit (dropped) blocks behave as blocks of
    zeroes, so we refer to a block matrix in which at least one block is
    implicitly zero as a **block-sparse matrix**. Otherwise, we say the matrix
    is block-dense. The property :meth:`is_sparse` encodes this state.

    Dropped blocks are not stored in memory or on :meth:`write`. In fact,
    blocks that are dropped prior to an action like :meth:`export` or
    :meth:`to_numpy` are never computed in the first place, nor are any blocks
    of upstream operands on which only dropped blocks depend! In addition,
    linear algebra is accelerated by avoiding, for example, explicit addition of
    or multiplication by blocks of zeroes.

    Block-sparse matrices may be created with
    :meth:`sparsify_band`,
    :meth:`sparsify_rectangles`,
    :meth:`sparsify_row_intervals`,
    and :meth:`sparsify_triangle`.

    The following methods naturally propagate block-sparsity:

    - Addition and subtraction "union" realized blocks.

    - Element-wise multiplication "intersects" realized blocks.

    - Transpose "transposes" realized blocks.

    - :meth:`abs` and :meth:`sqrt` preserve the realized blocks.

    - :meth:`sum` along an axis realizes those blocks for which at least one
      block summand is realized.

    - Matrix slicing, and more generally :meth:`filter`, :meth:`filter_rows`,
      and :meth:`filter_cols`.

    These following methods always result in a block-dense matrix:

    - :meth:`fill`

    - Addition or subtraction of a scalar or broadcasted vector.

    - Matrix multiplication, ``@``.

    The following methods fail if any operand is block-sparse, but can be forced
    by first applying :meth:`densify`.

    - Element-wise division between two block matrices.

    - Multiplication by a scalar or broadcasted vector which includes an
      infinite or ``nan`` value.

    - Division by a scalar or broadcasted vector which includes a zero, infinite
      or ``nan`` value.

    - Division of a scalar or broadcasted vector by a block matrix.

    - Element-wise exponentiation by a negative exponent.

    - Natural logarithm, :meth:`log`.
    """

    def __init__(self, bmir):
        self._bmir = bmir

    @classmethod
    @typecheck_method(path=str, _assert_type=nullable(tblockmatrix))
    def read(cls, path, *, _assert_type=None):
        """Reads a block matrix.

        Parameters
        ----------
        path: :class:`str`
            Path to input file.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return cls(BlockMatrixRead(BlockMatrixNativeReader(path), _assert_type=_assert_type))

    @classmethod
    @typecheck_method(uri=str,
                      n_rows=int,
                      n_cols=int,
                      block_size=nullable(int),
                      _assert_type=nullable(tblockmatrix))
    def fromfile(cls, uri, n_rows, n_cols, block_size=None, *, _assert_type=None):
        """Creates a block matrix from a binary file.

        Examples
        --------
        >>> import numpy as np
        >>> a = np.random.rand(10, 20)
        >>> a.tofile('/local/file') # doctest: +SKIP

        To create a block matrix of the same dimensions:

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
        uri: :class:`str`, optional
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

        return cls(BlockMatrixRead(BlockMatrixBinaryReader(uri, [n_rows, n_cols], block_size), _assert_type=_assert_type))

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
        from hail.backend.service_backend import ServiceBackend

        if not block_size:
            block_size = BlockMatrix.default_block_size()

        if any(i == 0 for i in ndarray.shape):
            raise ValueError(f'from_numpy: ndarray dimensions must be non-zero, found shape {ndarray.shape}')

        nd = _ndarray_as_2d(ndarray)
        nd = _ndarray_as_float64(nd)
        n_rows, n_cols = nd.shape

        if isinstance(hl.current_backend(), ServiceBackend):
            path = hl.TemporaryFilename().name
            hl.current_backend().fs.open(path, mode='wb').write(nd.tobytes())
            uri = path
        else:
            path = new_local_temp_file()
            nd.tofile(path)
            uri = local_path_uri(path)
        return cls.fromfile(uri, n_rows, n_cols, block_size)

    @classmethod
    @typecheck_method(entry_expr=expr_float64,
                      mean_impute=bool,
                      center=bool,
                      normalize=bool,
                      axis=nullable(enumeration('rows', 'cols')),
                      block_size=nullable(int))
    def from_entry_expr(cls, entry_expr, mean_impute=False, center=False, normalize=False, axis='rows', block_size=None):
        """Creates a block matrix using a matrix table entry expression.

        Examples
        --------
        >>> mt = hl.balding_nichols_model(3, 25, 50)
        >>> bm = BlockMatrix.from_entry_expr(mt.GT.n_alt_alleles())

        Notes
        -----
        This convenience method writes the block matrix to a temporary file on
        persistent disk and then reads the file. If you want to store the
        resulting block matrix, use :meth:`write_from_entry_expr` directly to
        avoid writing the result twice. See :meth:`write_from_entry_expr` for
        further documentation.

        Warning
        -------
        If the rows of the matrix table have been filtered to a small fraction,
        then :meth:`.MatrixTable.repartition` before this method to improve
        performance.

        If you encounter a Hadoop write/replication error, increase the
        number of persistent workers or the disk size per persistent worker,
        or use :meth:`write_from_entry_expr` to write to external storage.

        This method opens ``n_cols / block_size`` files concurrently per task.
        To not blow out memory when the number of columns is very large,
        limit the Hadoop write buffer size; e.g. on GCP, set this property on
        cluster startup (the default is 64MB):
        ``--properties 'core:fs.gs.io.buffersize.write=1048576``.

        Parameters
        ----------
        entry_expr: :class:`.Float64Expression`
            Entry expression for numeric matrix entries.
        mean_impute: :obj:`bool`
            If true, set missing values to the row mean before centering or
            normalizing. If false, missing values will raise an error.
        center: :obj:`bool`
            If true, subtract the row mean.
        normalize: :obj:`bool`
            If true and ``center=False``, divide by the row magnitude.
            If true and ``center=True``, divide the centered value by the
            centered row magnitude.
        axis: :class:`str`
            One of "rows" or "cols": axis by which to normalize or center.
        block_size: :obj:`int`, optional
            Block size. Default given by :meth:`.BlockMatrix.default_block_size`.
        """
        path = new_temp_file()
        cls.write_from_entry_expr(entry_expr, path, overwrite=False, mean_impute=mean_impute,
                                  center=center, normalize=normalize, axis=axis, block_size=block_size)
        return cls.read(path)

    @classmethod
    @typecheck_method(n_rows=int,
                      n_cols=int,
                      block_size=nullable(int),
                      seed=nullable(int),
                      gaussian=bool)
    def random(cls, n_rows, n_cols, block_size=None, seed=None, gaussian=True) -> 'BlockMatrix':
        """Creates a block matrix with standard normal or uniform random entries.

        Examples
        --------
        Create a block matrix with 10 rows, 20 columns, and standard normal entries:

        >>> bm = BlockMatrix.random(10, 20)

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
        gaussian: :obj:`bool`
            If ``True``, entries are drawn from the standard
            normal distribution. If ``False``, entries are drawn from
            the uniform distribution on [0,1].

        Returns
        -------
        :class:`.BlockMatrix`
        """
        if not block_size:
            block_size = BlockMatrix.default_block_size()

        static_rng_uid = seed if seed is not None else Env.next_static_rng_uid()

        rand = BlockMatrixRandom(static_rng_uid, gaussian, [n_rows, n_cols], block_size)
        return BlockMatrix(rand)

    @classmethod
    @typecheck_method(n_rows=int,
                      n_cols=int,
                      value=numeric,
                      block_size=nullable(int))
    def fill(cls, n_rows, n_cols, value, block_size=None):
        """Creates a block matrix with all elements the same value.

        Examples
        --------
        Create a block matrix with 10 rows, 20 columns, and all elements equal to ``1.0``:

        >>> bm = BlockMatrix.fill(10, 20, 1.0)

        Parameters
        ----------
        n_rows: :obj:`int`
            Number of rows.
        n_cols: :obj:`int`
            Number of columns.
        value: :obj:`float`
            Value of all elements.
        block_size: :obj:`int`, optional
            Block size. Default given by :meth:`default_block_size`.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        if not block_size:
            block_size = BlockMatrix.default_block_size()

        bmir = BlockMatrixBroadcast(_to_bmir(value, block_size),
                                    [], [n_rows, n_cols],
                                    block_size)
        return BlockMatrix(bmir)

    @classmethod
    @typecheck_method(n_rows=int,
                      n_cols=int,
                      data=sequenceof(float),
                      block_size=nullable(int))
    def _create(cls, n_rows, n_cols, data, block_size=None):
        """Private method for creating small test matrices."""

        if block_size is None:
            block_size = BlockMatrix.default_block_size()

        return BlockMatrix(ValueToBlockMatrix(hl.literal(data)._ir, [n_rows, n_cols], block_size))

    @classmethod
    @typecheck_method(ndarray_expression=expr_ndarray(), block_size=int)
    def from_ndarray(cls, ndarray_expression, block_size=4096):
        """Create a BlockMatrix from an ndarray"""
        if ndarray_expression.dtype.element_type != hl.tfloat64:
            raise ValueError("BlockMatrix.from_ndarray expects an ndarray of type float64")

        shape = hl.eval(ndarray_expression.shape)

        if shape is None:
            raise ValueError("Cannot make a BlockMatrix from a missing NDArray")
        return BlockMatrix(ValueToBlockMatrix(ndarray_expression._ir, shape, block_size))

    @staticmethod
    def default_block_size():
        """Default block side length."""

        # This should match BlockMatrix.defaultBlockSize in the Scala backend.
        return 4096  # 32 * 1024 bytes

    @property
    def element_type(self):
        """The type of the elements"""
        return self._bmir.typ.element_type

    @property
    def n_rows(self):
        """Number of rows.

        Returns
        -------
        :obj:`int`
        """
        return self.shape[0]

    @property
    def n_cols(self):
        """Number of columns.

        Returns
        -------
        :obj:`int`
        """
        return self.shape[1]

    @property
    def _n_block_rows(self):
        return (self.n_rows + self.block_size - 1) // self.block_size

    @property
    def _n_block_cols(self):
        return (self.n_cols + self.block_size - 1) // self.block_size

    @property
    def shape(self):
        """Shape of matrix.

        Returns
        -------
        (:obj:`int`, :obj:`int`)
           Number of rows and number of columns.
        """
        return tensor_shape_to_matrix_shape(self._bmir)

    @property
    def block_size(self):
        """Block size.

        Returns
        -------
        :obj:`int`
        """
        return self._bmir.typ.block_size

    @property
    def _last_col_block_width(self):
        remainder = self.n_cols % self.block_size
        return remainder if remainder != 0 else self.block_size

    @property
    def _last_row_block_height(self):
        remainder = self.n_rows % self.block_size
        return remainder if remainder != 0 else self.block_size

    @typecheck_method(path=str,
                      overwrite=bool,
                      force_row_major=bool,
                      stage_locally=bool)
    def write(self, path, overwrite=False, force_row_major=False, stage_locally=False):
        """Writes the block matrix.

        .. include:: ../_templates/write_warning.rst

        Parameters
        ----------
        path: :class:`str`
            Path for output file.
        overwrite : :obj:`bool`
            If ``True``, overwrite an existing file at the destination.
        force_row_major: :obj:`bool`
            If ``True``, transform blocks in column-major format
            to row-major format before writing.
            If ``False``, write blocks in their current format.
        stage_locally: :obj:`bool`
            If ``True``, major output will be written to temporary local storage
            before being copied to ``output``.
        """
        hl.current_backend().validate_file(path)

        writer = BlockMatrixNativeWriter(path, overwrite, force_row_major, stage_locally)
        Env.backend().execute(BlockMatrixWrite(self._bmir, writer))

    @typecheck_method(path=str,
                      overwrite=bool,
                      force_row_major=bool,
                      stage_locally=bool)
    def checkpoint(self, path, overwrite=False, force_row_major=False, stage_locally=False):
        """Checkpoint the block matrix.

        .. include:: ../_templates/write_warning.rst

        Parameters
        ----------
        path: :class:`str`
            Path for output file.
        overwrite : :obj:`bool`
            If ``True``, overwrite an existing file at the destination.
        force_row_major: :obj:`bool`
            If ``True``, transform blocks in column-major format
            to row-major format before checkpointing.
            If ``False``, checkpoint blocks in their current format.
        stage_locally: :obj:`bool`
            If ``True``, major output will be written to temporary local storage
            before being copied to ``output``.
        """
        hl.current_backend().validate_file(path)
        self.write(path, overwrite, force_row_major, stage_locally)
        return BlockMatrix.read(path, _assert_type=self._bmir._type)

    @staticmethod
    @typecheck(entry_expr=expr_float64,
               path=str,
               overwrite=bool,
               mean_impute=bool,
               center=bool,
               normalize=bool,
               axis=nullable(enumeration('rows', 'cols')),
               block_size=nullable(int))
    def write_from_entry_expr(entry_expr, path, overwrite=False, mean_impute=False,
                              center=False, normalize=False, axis='rows', block_size=None):
        """Writes a block matrix from a matrix table entry expression.

        Examples
        --------
        >>> mt = hl.balding_nichols_model(3, 25, 50)
        >>> BlockMatrix.write_from_entry_expr(mt.GT.n_alt_alleles(),
        ...                                   'output/model.bm')

        Notes
        -----
        The resulting file can be loaded with :meth:`BlockMatrix.read`.
        Blocks are stored row-major.

        If a pipelined transformation significantly downsamples the rows of the
        underlying matrix table, then repartitioning the matrix table ahead of
        this method will greatly improve its performance.

        By default, this method will fail if any values are missing (to be clear,
        special float values like ``nan`` are not missing values).

        - Set `mean_impute` to replace missing values with the row mean before
          possibly centering or normalizing. If all values are missing, the row
          mean is ``nan``.

        - Set `center` to shift each row to have mean zero before possibly
          normalizing.

        - Set `normalize` to normalize each row to have unit length.

        To standardize each row, regarded as an empirical distribution, to have
        mean 0 and variance 1, set `center` and `normalize` and then multiply
        the result by ``sqrt(n_cols)``.

        Warning
        -------
        If the rows of the matrix table have been filtered to a small fraction,
        then :meth:`.MatrixTable.repartition` before this method to improve
        performance.

        This method opens ``n_cols / block_size`` files concurrently per task.
        To not blow out memory when the number of columns is very large,
        limit the Hadoop write buffer size; e.g. on GCP, set this property on
        cluster startup (the default is 64MB):
        ``--properties 'core:fs.gs.io.buffersize.write=1048576``.

        Parameters
        ----------
        entry_expr: :class:`.Float64Expression`
            Entry expression for numeric matrix entries.
        path: :class:`str`
            Path for output.
        overwrite : :obj:`bool`
            If ``True``, overwrite an existing file at the destination.
        mean_impute: :obj:`bool`
            If true, set missing values to the row mean before centering or
            normalizing. If false, missing values will raise an error.
        center: :obj:`bool`
            If true, subtract the row mean.
        normalize: :obj:`bool`
            If true and ``center=False``, divide by the row magnitude.
            If true and ``center=True``, divide the centered value by the
            centered row magnitude.
        axis: :class:`str`
            One of "rows" or "cols": axis by which to normalize or center.
        block_size: :obj:`int`, optional
            Block size. Default given by :meth:`.BlockMatrix.default_block_size`.
        """
        hl.current_backend().validate_file(path)

        if not block_size:
            block_size = BlockMatrix.default_block_size()

        check_entry_indexed('BlockMatrix.write_from_entry_expr', entry_expr)
        mt = matrix_table_source('BlockMatrix.write_from_entry_expr', entry_expr)

        if not (mean_impute or center or normalize):
            if entry_expr in mt._fields_inverse:
                field = mt._fields_inverse[entry_expr]
                mt.select_entries(field)._write_block_matrix(path, overwrite, field, block_size)
            else:
                field = Env.get_uid()
                mt.select_entries(**{field: entry_expr})._write_block_matrix(path, overwrite, field, block_size)
        else:
            mt = mt.select_entries(__x=entry_expr).unfilter_entries()
            compute = {
                '__count': agg.count_where(hl.is_defined(mt['__x'])),
                '__sum': agg.sum(mt['__x']),
                '__sum_sq': agg.sum(mt['__x'] * mt['__x'])
            }
            if axis == 'rows':
                n_elements = mt.count_cols()
                mt = mt.select_rows(**compute)
            else:
                n_elements = mt.count_rows()
                mt = mt.select_cols(**compute)
            compute = {
                '__mean': mt['__sum'] / mt['__count'],
                '__centered_length': hl.sqrt(mt['__sum_sq']
                                             - (mt['__sum'] ** 2) / mt['__count']),
                '__length': hl.sqrt(mt['__sum_sq']
                                    + (n_elements - mt['__count'])
                                    * ((mt['__sum'] / mt['__count']) ** 2))
            }
            if axis == 'rows':
                mt = mt.select_rows(**compute)
            else:
                mt = mt.select_cols(**compute)
            expr = mt['__x']
            if normalize:
                if center:
                    expr = (expr - mt['__mean']) / mt['__centered_length']
                    if mean_impute:
                        expr = hl.or_else(expr, 0.0)
                else:
                    if mean_impute:
                        expr = hl.or_else(expr, mt['__mean'])
                    expr = expr / mt['__length']
            else:
                if center:
                    expr = expr - mt['__mean']
                    if mean_impute:
                        expr = hl.or_else(expr, 0.0)
                else:
                    if mean_impute:
                        expr = hl.or_else(expr, mt['__mean'])

            field = Env.get_uid()
            mt.select_entries(**{field: expr})._write_block_matrix(path, overwrite, field, block_size)

    @staticmethod
    def _check_indices(indices, size):
        if len(indices) == 0:
            raise ValueError('index list must be non-empty')
        elif not all(x < y for x, y in zip(indices, indices[1:])):
            raise ValueError('index list must be strictly increasing')
        elif indices[0] < 0:
            raise ValueError(f'index list values must be in range [0, {size}), found {indices[0]}')
        elif indices[-1] >= size:
            raise ValueError(f'index list values must be in range [0, {size}), found {indices[-1]}')

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
        BlockMatrix._check_indices(rows_to_keep, self.n_rows)
        return BlockMatrix(BlockMatrixFilter(self._bmir, [rows_to_keep, []]))

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
        BlockMatrix._check_indices(cols_to_keep, self.n_cols)
        return BlockMatrix(BlockMatrixFilter(self._bmir, [[], cols_to_keep]))

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
        BlockMatrix._check_indices(rows_to_keep, self.n_rows)
        BlockMatrix._check_indices(cols_to_keep, self.n_cols)
        return BlockMatrix(BlockMatrixFilter(self._bmir, [rows_to_keep, cols_to_keep]))

    @staticmethod
    def _pos_index(i, size, name, allow_size=False):
        if 0 <= i < size or (i == size and allow_size):
            return i
        elif 0 <= i + size < size:
            return i + size
        else:
            raise ValueError(f'invalid {name} {i} for axis of size {size}')

    @staticmethod
    def _range_to_keep(idx, size):
        if isinstance(idx, int):
            pos_idx = BlockMatrix._pos_index(idx, size, 'index')
            return slice(pos_idx, pos_idx + 1, 1)

        assert isinstance(idx, slice)
        if idx.step and idx.step <= 0:
            raise ValueError(f'slice step must be positive, found {idx.step}')

        start = 0 if idx.start is None else BlockMatrix._pos_index(idx.start, size, 'start index')
        stop = size if idx.stop is None else BlockMatrix._pos_index(idx.stop, size, 'stop index', allow_size=True)
        step = 1 if idx.step is None else idx.step

        if start < stop:
            return slice(start, stop, step)
        else:
            raise ValueError(f'slice {start}:{stop}:{step} is empty')

    @typecheck_method(indices=tupleof(oneof(int, sliceof(nullable(int), nullable(int), nullable(int)))))
    def __getitem__(self, indices):
        if len(indices) != 2:
            raise ValueError(f'tuple of indices or slices must have length two, found {len(indices)}')

        row_idx, col_idx = indices

        if isinstance(row_idx, int) and isinstance(col_idx, int):
            i = BlockMatrix._pos_index(row_idx, self.n_rows, 'row index')
            j = BlockMatrix._pos_index(col_idx, self.n_cols, 'col index')

            return Env.backend().execute(BlockMatrixToValueApply(self._bmir,
                                                                 {'name': 'GetElement', 'index': [i, j]}))

        rows_to_keep = BlockMatrix._range_to_keep(row_idx, self.n_rows)
        cols_to_keep = BlockMatrix._range_to_keep(col_idx, self.n_cols)

        return BlockMatrix(BlockMatrixSlice(self._bmir, [rows_to_keep, cols_to_keep]))

    @typecheck_method(lower=int, upper=int, blocks_only=bool)
    def sparsify_band(self, lower=0, upper=0, blocks_only=False):
        r"""Filter to a diagonal band.

        Examples
        --------
        Consider the following block matrix:

        >>> import numpy as np
        >>> nd = np.array([[ 1.0,  2.0,  3.0,  4.0],
        ...                [ 5.0,  6.0,  7.0,  8.0],
        ...                [ 9.0, 10.0, 11.0, 12.0],
        ...                [13.0, 14.0, 15.0, 16.0]])
        >>> bm = BlockMatrix.from_numpy(nd, block_size=2)

        Filter to a band from one below the diagonal to
        two above the diagonal and collect to NumPy:

        >>> bm.sparsify_band(lower=-1, upper=2).to_numpy()  # doctest: +SKIP_OUTPUT_CHECK
        array([[ 1.,  2.,  3.,  0.],
               [ 5.,  6.,  7.,  8.],
               [ 0., 10., 11., 12.],
               [ 0.,  0., 15., 16.]])

        Set all blocks fully outside the diagonal to zero
        and collect to NumPy:

        >>> bm.sparsify_band(lower=0, upper=0, blocks_only=True).to_numpy()  # doctest: +SKIP_OUTPUT_CHECK
        array([[ 1.,  2.,  0.,  0.],
               [ 5.,  6.,  0.,  0.],
               [ 0.,  0., 11., 12.],
               [ 0.,  0., 15., 16.]])

        Notes
        -----
        This method creates a block-sparse matrix by zeroing out all blocks
        which are disjoint from a diagonal band. By default,
        all elements outside the band but inside blocks that overlap the
        band are set to zero as well.

        The band is defined in terms of inclusive `lower` and `upper` indices
        relative to the diagonal. For example, the indices -1, 0, and 1
        correspond to the sub-diagonal, diagonal, and super-diagonal,
        respectively. The diagonal band contains the elements at positions
        :math:`(i, j)` such that

        .. math::

          \mathrm{lower} \leq j - i \leq \mathrm{upper}.

        `lower` must be less than or equal to `upper`, but their values may
        exceed the dimensions of the matrix, the band need not include the
        diagonal, and the matrix need not be square.

        Parameters
        ----------
        lower: :obj:`int`
            Index of lowest band relative to the diagonal.
        upper: :obj:`int`
            Index of highest band relative to the diagonal.
        blocks_only: :obj:`bool`
            If ``False``, set all elements outside the band to zero.
            If ``True``, only set all blocks outside the band to blocks
            of zeros; this is more efficient.

        Returns
        -------
        :class:`.BlockMatrix`
            Sparse block matrix.
        """
        if lower > upper:
            raise ValueError(f'sparsify_band: lower={lower} is greater than upper={upper}')

        bounds = hl.literal((lower, upper), hl.ttuple(hl.tint64, hl.tint64))
        return BlockMatrix(BlockMatrixSparsify(self._bmir, bounds._ir, BandSparsifier(blocks_only)))

    @typecheck_method(lower=bool, blocks_only=bool)
    def sparsify_triangle(self, lower=False, blocks_only=False):
        """Filter to the upper or lower triangle.

        Examples
        --------
        Consider the following block matrix:

        >>> import numpy as np
        >>> nd = np.array([[ 1.0,  2.0,  3.0,  4.0],
        ...                [ 5.0,  6.0,  7.0,  8.0],
        ...                [ 9.0, 10.0, 11.0, 12.0],
        ...                [13.0, 14.0, 15.0, 16.0]])
        >>> bm = BlockMatrix.from_numpy(nd, block_size=2)

        Filter to the upper triangle and collect to NumPy:

        >>> bm.sparsify_triangle().to_numpy()  # doctest: +SKIP_OUTPUT_CHECK
        array([[ 1.,  2.,  3.,  4.],
               [ 0.,  6.,  7.,  8.],
               [ 0.,  0., 11., 12.],
               [ 0.,  0.,  0., 16.]])

        Set all blocks fully outside the upper triangle to zero
        and collect to NumPy:

        >>> bm.sparsify_triangle(blocks_only=True).to_numpy()  # doctest: +SKIP_OUTPUT_CHECK
        array([[ 1.,  2.,  3.,  4.],
               [ 5.,  6.,  7.,  8.],
               [ 0.,  0., 11., 12.],
               [ 0.,  0., 15., 16.]])

        Notes
        -----
        This method creates a block-sparse matrix by zeroing out all blocks
        which are disjoint from the (non-strict) upper or lower triangle. By
        default, all elements outside the triangle but inside blocks that
        overlap the triangle are set to zero as well.

        Parameters
        ----------
        lower: :obj:`bool`
            If ``False``, keep the upper triangle.
            If ``True``, keep the lower triangle.
        blocks_only: :obj:`bool`
            If ``False``, set all elements outside the triangle to zero.
            If ``True``, only set all blocks outside the triangle to
            blocks of zeros; this is more efficient.

        Returns
        -------
        :class:`.BlockMatrix`
            Sparse block matrix.
        """
        if lower:
            lower_band = 1 - self.n_rows
            upper_band = 0
        else:
            lower_band = 0
            upper_band = self.n_cols - 1

        return self.sparsify_band(lower_band, upper_band, blocks_only)

    @typecheck_method(intervals=expr_tuple([expr_array(expr_int64), expr_array(expr_int64)]),
                      blocks_only=bool)
    def _sparsify_row_intervals_expr(self, intervals, blocks_only=False):
        return BlockMatrix(
            BlockMatrixSparsify(self._bmir, intervals._ir,
                                RowIntervalSparsifier(blocks_only)))

    @typecheck_method(indices=expr_array(expr_int32))
    def _sparsify_blocks(self, indices):
        return BlockMatrix(
            BlockMatrixSparsify(self._bmir, indices._ir,
                                PerBlockSparsifier()))

    @typecheck_method(starts=oneof(sequenceof(int), np.ndarray),
                      stops=oneof(sequenceof(int), np.ndarray),
                      blocks_only=bool)
    def sparsify_row_intervals(self, starts, stops, blocks_only=False):
        """Creates a block-sparse matrix by filtering to an interval for each row.

        Examples
        --------
        Consider the following block matrix:

        >>> import numpy as np
        >>> nd = np.array([[ 1.0,  2.0,  3.0,  4.0],
        ...                [ 5.0,  6.0,  7.0,  8.0],
        ...                [ 9.0, 10.0, 11.0, 12.0],
        ...                [13.0, 14.0, 15.0, 16.0]])
        >>> bm = BlockMatrix.from_numpy(nd, block_size=2)

        Set all elements outside the given row intervals to zero
        and collect to NumPy:

        >>> (bm.sparsify_row_intervals(starts=[1, 0, 2, 2],
        ...                            stops= [2, 0, 3, 4])
        ...    .to_numpy())  # doctest: +SKIP_OUTPUT_CHECK
        array([[ 0.,  2.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0., 11.,  0.],
               [ 0.,  0., 15., 16.]])

        Set all blocks fully outside the given row intervals to
        blocks of zeros and collect to NumPy:

        >>> (bm.sparsify_row_intervals(starts=[1, 0, 2, 2],
        ...                            stops= [2, 0, 3, 4],
        ...                            blocks_only=True)
        ...    .to_numpy())  # doctest: +SKIP_OUTPUT_CHECK
        array([[ 1.,  2.,  0.,  0.],
               [ 5.,  6.,  0.,  0.],
               [ 0.,  0., 11., 12.],
               [ 0.,  0., 15., 16.]])

        Notes
        -----
        This method creates a block-sparse matrix by zeroing out all blocks
        which are disjoint from all row intervals. By default, all elements
        outside the row intervals but inside blocks that overlap the row
        intervals are set to zero as well.

        `starts` and `stops` must both have length equal to the number of
        rows. The interval for row ``i`` is ``[starts[i], stops[i])``. In
        particular, ``0 <= starts[i] <= stops[i] <= n_cols`` is required
        for all ``i``.

        This method requires the number of rows to be less than :math:`2^{31}`.

        Parameters
        ----------
        starts: :obj:`list` of :obj:`int`, or :class:`numpy.ndarray` of :obj:`int`
            Start indices for each row (inclusive).
        stops: :obj:`list` of :obj:`int`, or :class:`numpy.ndarray` of :obj:`int`
            Stop indices for each row (exclusive).
        blocks_only: :obj:`bool`
            If ``False``, set all elements outside row intervals to zero.
            If ``True``, only set all blocks outside row intervals to blocks
            of zeros; this is more efficient.
        Returns
        -------
        :class:`.BlockMatrix`
            Sparse block matrix.
        """
        if isinstance(starts, np.ndarray):
            if not (starts.dtype == np.int32 or starts.dtype == np.int64):
                raise ValueError("sparsify_row_intervals: starts ndarray must have dtype 'int32' or 'int64'")
            starts = [int(s) for s in starts]
        if isinstance(stops, np.ndarray):
            if not (stops.dtype == np.int32 or stops.dtype == np.int64):
                raise ValueError("sparsify_row_intervals: stops ndarray must have dtype 'int32' or 'int64'")
            stops = [int(s) for s in stops]

        n_rows = self.n_rows
        n_cols = self.n_cols
        if n_rows >= (1 << 31):
            raise ValueError(f'n_rows must be less than 2^31, found {n_rows}')
        if len(starts) != n_rows or len(stops) != n_rows:
            raise ValueError(f'starts and stops must both have length {n_rows} (the number of rows)')
        if any([start < 0 for start in starts]):
            raise ValueError('all start values must be non-negative')
        if any([stop > self.n_cols for stop in stops]):
            raise ValueError(f'all stop values must be less than or equal to {n_cols} (the number of columns)')
        if any([starts[i] > stops[i] for i in range(0, n_rows)]):
            raise ValueError('every start value must be less than or equal to the corresponding stop value')

        return self._sparsify_row_intervals_expr((starts, stops), blocks_only)

    @typecheck_method(uri=str)
    def tofile(self, uri):
        """Collects and writes data to a binary file.

        Examples
        --------
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
        uri: :class:`str`, optional
            URI of binary output file.

        See Also
        --------
        :meth:`.to_numpy`
        """
        hl.current_backend().validate_file(uri)

        _check_entries_size(self.n_rows, self.n_cols)

        writer = BlockMatrixBinaryWriter(uri)
        Env.backend().execute(BlockMatrixWrite(self._bmir, writer))

    @typecheck_method(_force_blocking=bool)
    def to_numpy(self, _force_blocking=False):
        """Collects the block matrix into a `NumPy ndarray
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`__.

        Examples
        --------
        >>> bm = BlockMatrix.random(10, 20)
        >>> a = bm.to_numpy()

        Notes
        -----
        The resulting ndarray will have the same shape as the block matrix.

        Returns
        -------
        :class:`numpy.ndarray`
        """
        from hail.backend.service_backend import ServiceBackend

        if self.n_rows * self.n_cols > 1 << 31 or _force_blocking:
            path = new_temp_file()
            self.export_blocks(path, binary=True)
            return BlockMatrix.rectangles_to_numpy(path, binary=True)

        if isinstance(hl.current_backend(), ServiceBackend):
            with hl.TemporaryFilename() as path:
                self.tofile(path)
                return np.frombuffer(
                    hl.current_backend().fs.open(path, mode='rb').read()
                ).reshape((self.n_rows, self.n_cols))

        with with_local_temp_file() as path:
            uri = local_path_uri(path)
            self.tofile(uri)
            return np.fromfile(path).reshape((self.n_rows, self.n_cols))

    def to_ndarray(self):
        """Collects a BlockMatrix into a local hail ndarray expression on driver. This should not
        be done for large matrices.

        Returns
        -------
        :class:`.NDArrayExpression`
        """
        ir = BlockMatrixCollect(self._bmir)
        return construct_expr(ir, hl.tndarray(hl.tfloat64, 2))

    @property
    def is_sparse(self):
        """Returns ``True`` if block-sparse.

        Notes
        -----
        A block matrix is block-sparse if at least of its blocks is dropped,
        i.e. implicitly a block of zeros.

        Returns
        -------
        :obj:`bool`
        """
        return Env.backend()._to_java_blockmatrix_ir(self._bmir).typ().isSparse()

    @property
    def T(self):
        """Matrix transpose.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        if self.n_rows == 1 and self.n_cols == 1:
            return self

        if self.n_rows == 1:
            index_expr = [0]
        elif self.n_cols == 1:
            index_expr = [1]
        else:
            index_expr = [1, 0]

        return BlockMatrix(BlockMatrixBroadcast(self._bmir, index_expr, [self.n_cols, self.n_rows], self.block_size))

    def densify(self):
        """Restore all dropped blocks as explicit blocks of zeros.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return BlockMatrix(BlockMatrixDensify(self._bmir))

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
        return Env.backend().persist_blockmatrix(self)

    def unpersist(self):
        """Unpersists this block matrix from memory/disk.

        Notes
        -----
        This function will have no effect on a block matrix that was not previously
        persisted.

        Returns
        -------
        :class:`.BlockMatrix`
            Unpersisted block matrix.
        """
        return Env.backend().unpersist_blockmatrix(self)

    def __pos__(self):
        return self

    def __neg__(self):
        """Negation: -a.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return self._apply_map(lambda x: construct_expr(ApplyUnaryPrimOp('-', x._ir), hl.tfloat64), needs_dense=False)

    @staticmethod
    def _binary_op(op):
        return lambda l, r: construct_expr(ApplyBinaryPrimOp(op, l._ir, r._ir), hl.tfloat64)

    @typecheck_method(f=func_spec(1, expr_float64), needs_dense=bool)
    def _apply_map(self, f, needs_dense):
        uid = Env.get_uid()
        bmir = self._bmir
        if needs_dense:
            bmir = BlockMatrixDensify(bmir)
        return BlockMatrix(BlockMatrixMap(bmir, uid, f(construct_variable(uid, hl.tfloat64))._ir, needs_dense))

    @typecheck_method(f=func_spec(2, expr_float64),
                      other=oneof(numeric, np.ndarray, block_matrix_type),
                      sparsity_strategy=str,
                      reverse=bool)
    def _apply_map2(self, f, other, sparsity_strategy, reverse=False):
        if not isinstance(other, BlockMatrix):
            other = BlockMatrix(_to_bmir(other, self.block_size))

        self_shape, other_shape = list(self.shape), list(other.shape)
        result_shape = _shape_after_broadcast(self_shape, other_shape)

        self_bmir = self._bmir if self_shape == result_shape else _broadcast_to_shape(self._bmir, result_shape)
        other_bmir = other._bmir if other_shape == result_shape else _broadcast_to_shape(other._bmir, result_shape)

        if reverse:
            left, right = other_bmir, self_bmir
        else:
            left, right = self_bmir, other_bmir

        lv = Env.get_uid()
        rv = Env.get_uid()
        f_ir = f(construct_variable(lv, hl.tfloat64), construct_variable(rv, hl.tfloat64))._ir
        return BlockMatrix(BlockMatrixMap2(left, right, lv, rv, f_ir, sparsity_strategy))

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
        if isinstance(b, (int, float)):
            return self._map_dense(lambda entry: entry + b)
        return self._apply_map2(BlockMatrix._binary_op('+'), b, sparsity_strategy="Union")

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
        if isinstance(b, (int, float)):
            return self._map_dense(lambda entry: entry - b)
        return self._apply_map2(BlockMatrix._binary_op('-'), b, sparsity_strategy="Union")

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
        if isinstance(b, (int, float)):
            # sparse since multiplying by zero is zero
            return self._map_sparse(lambda entry: entry * b)
        return self._apply_map2(BlockMatrix._binary_op('*'), b, sparsity_strategy="Intersection")

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
        if isinstance(b, (int, float)):
            # sparse since dividing by zero is zero
            return self._map_sparse(lambda entry: entry / b)
        return self._apply_map2(BlockMatrix._binary_op('/'), b, sparsity_strategy="NeedsDense")

    @typecheck_method(b=numeric)
    def __radd__(self, b):
        return self._apply_map2(BlockMatrix._binary_op('+'), b, sparsity_strategy="Union", reverse=True)

    @typecheck_method(b=numeric)
    def __rsub__(self, b):
        return self._apply_map2(BlockMatrix._binary_op('-'), b, sparsity_strategy="Union", reverse=True)

    @typecheck_method(b=numeric)
    def __rmul__(self, b):
        return self._apply_map2(BlockMatrix._binary_op('*'), b, sparsity_strategy="Intersection", reverse=True)

    @typecheck_method(b=numeric)
    def __rtruediv__(self, b):
        return self._apply_map2(BlockMatrix._binary_op('/'), b, sparsity_strategy="NeedsDense", reverse=True)

    @typecheck_method(block_row_range=sized_tupleof(int, int), block_col_range=sized_tupleof(int, int))
    def _select_blocks(self, block_row_range, block_col_range):
        start_brow, stop_brow = block_row_range
        start_bcol, stop_bcol = block_col_range

        start_row = start_brow * self.block_size
        stop_row = (stop_brow - 1) * self.block_size + (self._last_row_block_height if stop_brow == self._n_block_rows else self.block_size)

        start_col = start_bcol * self.block_size
        stop_col = (stop_bcol - 1) * self.block_size + (self._last_col_block_width if stop_bcol == self._n_block_cols else self.block_size)

        return self[start_row:stop_row, start_col:stop_col]

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
            b = BlockMatrix(_to_bmir(b, self.block_size))

        if self.n_cols != b.n_rows:
            raise ValueError(f'incompatible shapes for matrix multiplication: {self.shape} and {b.shape}')

        return BlockMatrix(BlockMatrixDot(self._bmir, b._bmir))

    @typecheck_method(b=oneof(np.ndarray, block_matrix_type), splits=int, path_prefix=nullable(str))
    def tree_matmul(self, b, *, splits, path_prefix=None):
        """Matrix multiplication in situations with large inner dimension.

        This function splits a single matrix multiplication into `split_on_inner` smaller matrix multiplications,
        does the smaller multiplications, checkpoints them with names defined by `file_name_prefix`, and adds them
        together. This is useful in cases when the multiplication of two large matrices results in a much smaller matrix.

        Parameters
        ----------
        b: :class:`numpy.ndarray` or :class:`BlockMatrix`
        splits: :obj:`int` (keyword only argument)
            The number of smaller multiplications to do.
        path_prefix: :class:`str` (keyword only argument)
            The prefix of the path to write the block matrices to. If unspecified, writes to a tmpdir.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        if isinstance(b, np.ndarray):
            b = BlockMatrix(_to_bmir(b, self.block_size))

        if self.n_cols != b.n_rows:
            raise ValueError(f'incompatible shapes for matrix multiplication: {self.shape} and {b.shape}')

        if path_prefix is None:
            path_prefix = new_temp_file("tree_matmul_tmp")

        if splits != 1:
            inner_brange_size = int(math.ceil(self._n_block_cols / splits))
            split_points = list(range(0, self._n_block_cols, inner_brange_size)) + [self._n_block_cols]
            inner_ranges = list(zip(split_points[:-1], split_points[1:]))
            blocks_to_multiply = [(self._select_blocks((0, self._n_block_rows), (start, stop)),
                                   b._select_blocks((start, stop), (0, b._n_block_cols))) for start, stop in inner_ranges]

            intermediate_multiply_exprs = [b1 @ b2 for b1, b2 in blocks_to_multiply]

            hl.experimental.write_block_matrices(intermediate_multiply_exprs, path_prefix)
            read_intermediates = [BlockMatrix.read(f"{path_prefix}_{i}") for i in range(0, len(intermediate_multiply_exprs))]

            return sum(read_intermediates)

        return BlockMatrix(BlockMatrixDot(self._bmir, b._bmir))

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
        return self._apply_map(lambda i: i ** x, needs_dense=False)

    def _map_dense(self, func):
        return self._apply_map(func, True)

    def _map_sparse(self, func):
        return self._apply_map(func, False)

    def sqrt(self):
        """Element-wise square root.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return self._apply_map(hl.sqrt, needs_dense=False)

    def ceil(self):
        """Element-wise ceiling.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return self._apply_map(hl.ceil, needs_dense=False)

    def floor(self):
        """Element-wise floor.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return self._apply_map(hl.floor, needs_dense=False)

    def abs(self):
        """Element-wise absolute value.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return self._apply_map(hl.abs, needs_dense=False)

    def log(self):
        """Element-wise natural logarithm.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return self._apply_map(lambda x: hl.log(x), needs_dense=True)

    def diagonal(self):
        """Extracts diagonal elements as a row vector.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        diag_bmir = BlockMatrixBroadcast(self._bmir,
                                         [0, 0],
                                         [1, min(self.n_rows, self.n_cols)],
                                         self.block_size)
        return BlockMatrix(diag_bmir)

    @typecheck_method(axis=nullable(int))
    def sum(self, axis=None):
        """Sums array elements over one or both axes.

        Examples
        --------
        >>> import numpy as np
        >>> nd = np.array([[ 1.0,  2.0,  3.0],
        ...                [ 4.0,  5.0,  6.0]])
        >>> bm = BlockMatrix.from_numpy(nd)
        >>> bm.sum()
        21.0

        >>> bm.sum(axis=0).to_numpy()
        array([[5., 7., 9.]])

        >>> bm.sum(axis=1).to_numpy()
        array([[ 6.],
               [15.]])

        Parameters
        ----------
        axis: :obj:`int`, optional
            Axis over which to sum.
            By default, sum all elements.
            If ``0``, sum over rows.
            If ``1``, sum over columns.

        Returns
        -------
        :obj:`float` or :class:`BlockMatrix`
            If None, returns a float.
            If ``0``, returns a block matrix with a single row.
            If ``1``, returns a block matrix with a single column.
        """
        if axis is None:
            bmir = BlockMatrixAgg(self._bmir, [0, 1])
            return BlockMatrix(bmir)[0, 0]
        elif axis == 0 or axis == 1:
            out_index_expr = [axis]

            bmir = BlockMatrixAgg(self._bmir, out_index_expr)
            return BlockMatrix(bmir)
        else:
            raise ValueError(f'axis must be None, 0, or 1: found {axis}')

    def entries(self, keyed=True):
        """Returns a table with the indices and value of each block matrix entry.

        Examples
        --------
        >>> import numpy as np
        >>> block_matrix = BlockMatrix.from_numpy(np.array([[5, 7], [2, 8]]), 2)
        >>> entries_table = block_matrix.entries()
        >>> entries_table.show()
        +-------+-------+----------+
        |     i |     j |    entry |
        +-------+-------+----------+
        | int64 | int64 |  float64 |
        +-------+-------+----------+
        |     0 |     0 | 5.00e+00 |
        |     0 |     1 | 7.00e+00 |
        |     1 |     0 | 2.00e+00 |
        |     1 |     1 | 8.00e+00 |
        +-------+-------+----------+

        Notes
        -----
        The resulting table may be filtered, aggregated, and queried, but should only be
        directly exported to disk if the block matrix is very small.

        For block-sparse matrices, only realized blocks are included. To force inclusion
        of zeroes in dropped blocks, apply :meth:`densify` first.

        The resulting table has the following fields:

        - **i** (:py:data:`.tint64`, key field) -- Row index.

        - **j** (:py:data:`.tint64`, key field) -- Column index.

        - **entry** (:py:data:`.tfloat64`) -- Value of entry.

        Returns
        -------
        :class:`.Table`
            Table with a row for each entry.
        """
        t = Table(BlockMatrixToTable(self._bmir))
        if keyed:
            t = t.key_by('i', 'j')
        return t

    @typecheck_method(n_partitions=nullable(int), maximum_cache_memory_in_bytes=nullable(int))
    def to_table_row_major(self, n_partitions=None, maximum_cache_memory_in_bytes=None):
        """Returns a table where each row represents a row in the block matrix.

        The resulting table has the following fields:
            - **row_idx** (:py:data.`tint64`, key field) -- Row index
            - **entries** (:py:class:`.tarray` of :py:data:`.tfloat64`) -- Entries for the row

        Examples
        --------
        >>> import numpy as np
        >>> block_matrix = BlockMatrix.from_numpy(np.array([[1, 2], [3, 4], [5, 6]]), 2)
        >>> t = block_matrix.to_table_row_major()
        >>> t.show()
        +---------+---------------------+
        | row_idx | entries             |
        +---------+---------------------+
        |   int64 | array<float64>      |
        +---------+---------------------+
        |       0 | [1.00e+00,2.00e+00] |
        |       1 | [3.00e+00,4.00e+00] |
        |       2 | [5.00e+00,6.00e+00] |
        +---------+---------------------+

        Parameters
        ----------
        n_partitions : int or None
            Number of partitions of the table.
        maximum_cache_memory_in_bytes : int or None
            The amount of memory to reserve, per partition, to cache rows of the
            matrix in memory. This value must be at least large enough to hold
            one row of the matrix in memory. If this value is exactly the size of
            one row, then a partition makes a network request for every row of
            every block. Larger values reduce the number of network requests. If
            memory permits, setting this value to the size of one output
            partition permits one network request per block per partition.

        Notes
        -----
        Does not support block-sparse matrices.

        Returns
        -------
        :class:`.Table`
            Table where each row corresponds to a row in the block matrix.
        """
        path = new_temp_file()
        if maximum_cache_memory_in_bytes and maximum_cache_memory_in_bytes > (1 << 31) - 1:
            raise ValueError(
                f'maximum_cache_memory_in_bytes must be less than 2^31 -1, was: {maximum_cache_memory_in_bytes}')

        self.write(path, overwrite=True, force_row_major=True)
        reader = TableFromBlockMatrixNativeReader(path, n_partitions, maximum_cache_memory_in_bytes)
        return Table(TableRead(reader))

    @typecheck_method(n_partitions=nullable(int), maximum_cache_memory_in_bytes=nullable(int))
    def to_matrix_table_row_major(self, n_partitions=None, maximum_cache_memory_in_bytes=None):
        """Returns a matrix table with row key of `row_idx` and col key `col_idx`, whose
        entries are structs of a single field `element`.

        Parameters
        ----------
        n_partitions : int or None
            Number of partitions of the matrix table.
        maximum_cache_memory_in_bytes : int or None
            The amount of memory to reserve, per partition, to cache rows of the
            matrix in memory. This value must be at least large enough to hold
            one row of the matrix in memory. If this value is exactly the size of
            one row, then a partition makes a network request for every row of
            every block. Larger values reduce the number of network requests. If
            memory permits, setting this value to the size of one output
            partition permits one network request per block per partition.

        Notes
        -----
        Does not support block-sparse matrices.

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table where each entry corresponds to an entry in the block matrix.
        """
        t = self.to_table_row_major(n_partitions, maximum_cache_memory_in_bytes)
        t = t.transmute(entries=t.entries.map(lambda i: hl.struct(element=i)))
        t = t.annotate_globals(cols=hl.range(self.n_cols).map(lambda i: hl.struct(col_idx=hl.int64(i))))
        return t._unlocalize_entries('entries', 'cols', ['col_idx'])

    @staticmethod
    @typecheck(path_in=str,
               path_out=str,
               delimiter=str,
               header=nullable(str),
               add_index=bool,
               parallel=nullable(ExportType.checker),
               partition_size=nullable(int),
               entries=enumeration('full', 'lower', 'strict_lower', 'upper', 'strict_upper'))
    def export(path_in, path_out, delimiter='\t', header=None, add_index=False, parallel=None,
               partition_size=None, entries='full'):
        """Exports a stored block matrix as a delimited text file.

        Examples
        --------
        Consider the following matrix.

        >>> import numpy as np
        >>> nd = np.array([[1.0, 0.8, 0.7],
        ...                [0.8, 1.0 ,0.3],
        ...                [0.7, 0.3, 1.0]])
        >>> BlockMatrix.from_numpy(nd).write('output/example.bm', overwrite=True, force_row_major=True)

        Export the full matrix as a file with tab-separated values:

        >>> BlockMatrix.export('output/example.bm', 'output/example.tsv')

        Export the upper-triangle of the matrix as a block gzipped file of
        comma-separated values.

        >>> BlockMatrix.export(path_in='output/example.bm',
        ...                    path_out='output/example.csv.bgz',
        ...                    delimiter=',',
        ...                    entries='upper')

        Export the full matrix with row indices in parallel as a folder of
        gzipped files, each with a header line for columns ``idx``, ``A``,
        ``B``, and ``C``.

        >>> BlockMatrix.export(path_in='output/example.bm',
        ...                    path_out='output/example.gz',
        ...                    header='\t'.join(['idx', 'A', 'B', 'C']),
        ...                    add_index=True,
        ...                    parallel='header_per_shard',
        ...                    partition_size=2)

        This produces two compressed files which uncompress to:

        .. code-block:: text

            idx A   B   C
            0   1.0 0.8 0.7
            1   0.8 1.0 0.3

        .. code-block:: text

            idx A   B   C
            2   0.7 0.3 1.0

        Warning
        -------
        The block matrix must be stored in row-major format, as results
        from :meth:`.BlockMatrix.write` with ``force_row_major=True`` and from
        :meth:`.BlockMatrix.write_from_entry_expr`. Otherwise,
        :meth:`export` will fail.

        Notes
        -----
        The five options for `entries` are illustrated below.

        Full:

        .. code-block:: text

            1.0 0.8 0.7
            0.8 1.0 0.3
            0.7 0.3 1.0

        Lower triangle:

        .. code-block:: text

            1.0
            0.8 1.0
            0.7 0.3 1.0

        Strict lower triangle:

        .. code-block:: text

            0.8
            0.7 0.3

        Upper triangle:

        .. code-block:: text

            1.0 0.8 0.7
            1.0 0.3
            1.0

        Strict upper triangle:

        .. code-block:: text

            0.8 0.7
            0.3

        The number of columns must be less than :math:`2^{31}`.

        The number of partitions (file shards) exported equals the ceiling
        of ``n_rows / partition_size``. By default, there is one partition
        per row of blocks in the block matrix. The number of partitions
        should be at least the number of cores for efficient parallelism.
        Setting the partition size to an exact (rather than approximate)
        divisor or multiple of the block size reduces superfluous shuffling
        of data.

        If `parallel` is ``None``, these file shards are then serially
        concatenated by one core into one file, a slow process. See
        other options below.

        It is highly recommended to export large files with a ``.bgz`` extension,
        which will use a block gzipped compression codec. These files can be
        read natively with Python's ``gzip.open`` and R's ``read.table``.

        Parameters
        ----------
        path_in: :class:`str`
            Path to input block matrix, stored row-major on disk.
        path_out: :class:`str`
            Path for export.
            Use extension ``.gz`` for gzip or ``.bgz`` for block gzip.
        delimiter: :class:`str`
            Column delimiter.
        header: :class:`str`, optional
            If provided, `header` is prepended before the first row of data.
        add_index: :obj:`bool`
            If ``True``, add an initial column with the absolute row index.
        parallel: :class:`str`, optional
            If ``'header_per_shard'``, create a folder with one file per
            partition, each with a header if provided.
            If ``'separate_header'``, create a folder with one file per
            partition without a header; write the header, if provided, in
            a separate file.
            If ``None``, serially concatenate the header and all partitions
            into one file; export will be slower.
            If `header` is ``None`` then ``'header_per_shard'`` and
            ``'separate_header'`` are equivalent.
        partition_size: :obj:`int`, optional
            Number of rows to group per partition for export.
            Default given by block size of the block matrix.
        entries: :class:`str`
            Describes which entries to export. One of:
            ``'full'``, ``'lower'``, ``'strict_lower'``, ``'upper'``, ``'strict_upper'``.
        """
        hl.current_backend().validate_file(path_out)

        export_type = ExportType.default(parallel)

        Env.spark_backend('BlockMatrix.export')._jbackend.pyExportBlockMatrix(
            path_in, path_out, delimiter, header, add_index, export_type, partition_size, entries)

    @typecheck_method(rectangles=sequenceof(sequenceof(int)))
    def sparsify_rectangles(self, rectangles):
        """Filter to blocks overlapping the union of rectangular regions.

        Examples
        --------
        Consider the following block matrix:

        >>> import numpy as np
        >>> nd = np.array([[ 1.0,  2.0,  3.0,  4.0],
        ...                [ 5.0,  6.0,  7.0,  8.0],
        ...                [ 9.0, 10.0, 11.0, 12.0],
        ...                [13.0, 14.0, 15.0, 16.0]])
        >>> bm = BlockMatrix.from_numpy(nd, block_size=2)

        Filter to blocks covering three rectangles and collect to NumPy:

        >>> bm.sparsify_rectangles([[0, 1, 0, 1], [0, 3, 0, 2], [1, 2, 0, 4]]).to_numpy()  # doctest: +SKIP_OUTPUT_CHECK
        array([[ 1.,  2.,  3.,  4.],
               [ 5.,  6.,  7.,  8.],
               [ 9., 10.,  0.,  0.],
               [13., 14.,  0.,  0.]])

        Notes
        -----
        This method creates a block-sparse matrix by zeroing out (dropping)
        all blocks which are disjoint from the union of a set of rectangular
        regions. Partially overlapping blocks are *not* modified.

        Each rectangle is encoded as a list of length four of
        the form ``[row_start, row_stop, col_start, col_stop]``,
        where starts are inclusive and stops are exclusive.
        These must satisfy ``0 <= row_start <= row_stop <= n_rows`` and
        ``0 <= col_start <= col_stop <= n_cols``.

        For example ``[0, 2, 1, 3]`` corresponds to the row-index range
        ``[0, 2)`` and column-index range ``[1, 3)``, i.e. the elements at
        positions ``(0, 1)``, ``(0, 2)``, ``(1, 1)``, and ``(1, 2)``.

        The number of rectangles must be less than :math:`2^{29}`.

        Parameters
        ----------
        rectangles: :obj:`list` of :obj:`list` of :obj:`int`
            List of rectangles of the form
            ``[row_start, row_stop, col_start, col_stop]``.

        Returns
        -------
        :class:`.BlockMatrix`
            Sparse block matrix.
        """
        n_rectangles = len(rectangles)
        if n_rectangles >= (1 << 29):
            raise ValueError(f'number of rectangles must be less than 2^29, found {n_rectangles}')

        n_rows = self.n_rows
        n_cols = self.n_cols
        for r in rectangles:
            if len(r) != 4:
                raise ValueError(f'rectangle {r} does not have length 4')
            if not (0 <= r[0] <= r[1] <= n_rows and 0 <= r[2] <= r[3] <= n_cols):
                raise ValueError(f'rectangle {r} does not satisfy '
                                 f'0 <= r[0] <= r[1] <= n_rows and 0 <= r[2] <= r[3] <= n_cols')

        rectangles = hl.literal(list(itertools.chain(*rectangles)), hl.tarray(hl.tint64))
        return BlockMatrix(
            BlockMatrixSparsify(self._bmir, rectangles._ir, RectangleSparsifier))

    @typecheck_method(path_out=str,
                      rectangles=sequenceof(sequenceof(int)),
                      delimiter=str,
                      binary=bool)
    def export_rectangles(self, path_out, rectangles, delimiter='\t', binary=False):
        """Export rectangular regions from a block matrix to delimited text or binary files.

        Examples
        --------
        Consider the following block matrix:

        >>> import numpy as np
        >>> nd = np.array([[ 1.0,  2.0,  3.0,  4.0],
        ...                [ 5.0,  6.0,  7.0,  8.0],
        ...                [ 9.0, 10.0, 11.0, 12.0],
        ...                [13.0, 14.0, 15.0, 16.0]])

        Filter to the three rectangles and export as TSV files.

        >>> rectangles = [[0, 1, 0, 1], [0, 3, 0, 2], [1, 2, 0, 4]]
        >>>
        >>> (BlockMatrix.from_numpy(nd)
        ...     .export_rectangles('output/example.bm', rectangles))

        This produces three files in the example folder.

        The first file is ``rect-0_0-1-0-1``:

        .. code-block:: text

            1.0

        The second file is ``rect-1_0-3-0-2``:

        .. code-block:: text

            1.0 2.0
            5.0 6.0
            9.0 10.0

        The third file is ``rect-2_1-2-0-4``:

        .. code-block:: text

            5.0 6.0 7.0 8.0

        Notes
        -----
        This method exports rectangular regions of a stored block matrix
        to delimited text or binary files, in parallel by region.

        Each rectangle is encoded as a list of length four of
        the form ``[row_start, row_stop, col_start, col_stop]``,
        where starts are inclusive and stops are exclusive.
        These must satisfy ``0 <= row_start <= row_stop <= n_rows`` and
        ``0 <= col_start <= col_stop <= n_cols``.

        For example ``[0, 2, 1, 3]`` corresponds to the row-index range
        ``[0, 2)`` and column-index range ``[1, 3)``, i.e. the elements at
        positions ``(0, 1)``, ``(0, 2)``, ``(1, 1)``, and ``(1, 2)``.

        Each file name encodes the index of the rectangle in `rectangles`
        and the bounds as formatted in the example.

        The block matrix can be sparse provided all blocks overlapping
        the rectangles are present, i.e. this method does not currently
        support implicit zeros.

        If `binary` is true, each element is exported as 8 bytes, in row
        major order with no delimiting, new lines, or shape information. Such
        files can instantiate, for example, NumPy ndarrays using
        `fromfile <https://docs.scipy.org/doc/numpy/reference/generated/numpy.fromfile.html>`__
        and
        `reshape <https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html>`__.
        Note however that these binary files are not platform independent; in
        particular, no byte-order or data-type information is saved.

        The number of rectangles must be less than :math:`2^{29}`.

        Parameters
        ----------
        path_out: :class:`str`
            Path for folder of exported files.
        rectangles: :obj:`list` of :obj:`list` of :obj:`int`
            List of rectangles of the form
            ``[row_start, row_stop, col_start, col_stop]``.
        delimiter: :class:`str`
            Column delimiter.
        binary: :obj:`bool`
            If true, export elements as raw bytes in row major order.
        """
        n_rectangles = len(rectangles)
        if n_rectangles == 0:
            raise ValueError('no rectangles provided')
        if n_rectangles >= (1 << 29):
            raise ValueError(f'number of rectangles must be less than 2^29, found {n_rectangles}')

        for r in rectangles:
            if len(r) != 4:
                raise ValueError(f'rectangle {r} does not have length 4')
            if not (0 <= r[0] <= r[1] <= self.n_rows and 0 <= r[2] <= r[3] <= self.n_cols):
                raise ValueError(f'rectangle {r} does not satisfy '
                                 f'0 <= r[0] <= r[1] <= n_rows and 0 <= r[2] <= r[3] <= n_cols')

        writer = BlockMatrixRectanglesWriter(path_out, rectangles, delimiter, binary)
        Env.backend().execute(BlockMatrixWrite(self._bmir, writer))

    @typecheck_method(path_out=str, delimiter=str, binary=bool)
    def export_blocks(self, path_out, delimiter='\t', binary=False):
        """Export each block of the block matrix as its own delimited text or binary file.
        This is a special case of :meth:`.export_rectangles`

        Examples
        --------
        Consider the following block matrix:

        >>> import numpy as np
        >>> nd = np.array([[ 1.0, 2.0, 3.0],
        ...                [ 4.0, 5.0, 6.0],
        ...                [ 7.0, 8.0, 9.0]])

        >>> BlockMatrix.from_numpy(nd, block_size=2).export_blocks('output/example')

        This produces four files in the example folder.

        The first file is ``rect-0_0-2-0-2``:

        .. code-block:: text

            1.0 2.0
            4.0 5.0

        The second file is ``rect-1_0-2-2-3``:

        .. code-block:: text

            3.0
            6.0

        The third file is ``rect-2_2-3-0-2``:

        .. code-block:: text

            7.0 8.0

        And the fourth file is ``rect-3_3-4-3-4``:

        .. code-block:: text

            9.0

        Notes
        -----
        This method does not have any matrix size limitations.

        If exporting to binary files, note that they are not platform independent. No byte-order
        or data-type information is saved.

        See Also
        --------
        :meth:`.rectangles_to_numpy`

        Parameters
        ----------
        path_out: :class:`str`
            Path for folder of exported files.
        delimiter: :class:`str`
            Column delimiter.
        binary: :obj:`bool`
            If true, export elements as raw bytes in row major order.
        """
        def rows_in_block(block_row):
            if block_row == self._n_block_rows - 1:
                return self.n_rows - block_row * self.block_size
            return self.block_size

        def cols_in_block(block_col):
            if block_col == self._n_block_cols - 1:
                return self.n_cols - block_col * self.block_size
            return self.block_size

        def bounds(block_row, block_col):
            start_row = block_row * self.block_size
            start_col = block_col * self.block_size
            end_row = start_row + rows_in_block(block_row)
            end_col = start_col + cols_in_block(block_col)

            return [start_row, end_row, start_col, end_col]

        block_indices = itertools.product(range(self._n_block_rows), range(self._n_block_cols))
        rectangles = [bounds(block_row, block_col) for (block_row, block_col) in block_indices]

        self.export_rectangles(path_out, rectangles, delimiter, binary)

    @staticmethod
    @typecheck(path=str, binary=bool)
    def rectangles_to_numpy(path, binary=False):
        """Instantiates a NumPy ndarray from files of rectangles written out using
        :meth:`.export_rectangles` or :meth:`.export_blocks`. For any given
        dimension, the ndarray will have length equal to the upper bound of that dimension
        across the union of the rectangles. Entries not covered by any rectangle will be initialized to 0.

        Examples
        --------
        Consider the following:

        >>> import numpy as np
        >>> nd = np.array([[ 1.0, 2.0, 3.0],
        ...                [ 4.0, 5.0, 6.0],
        ...                [ 7.0, 8.0, 9.0]])

        >>> BlockMatrix.from_numpy(nd).export_rectangles('output/example', [[0, 3, 0, 1], [1, 2, 0, 2]])
        >>> BlockMatrix.rectangles_to_numpy('output/example')

        This would produce the following NumPy ndarray:

        .. code-block:: text

            1.0 0.0
            4.0 5.0
            7.0 0.0

        Notes
        -----
        If exporting to binary files, note that they are not platform independent. No byte-order
        or data-type information is saved.

        See Also
        --------
        :meth:`.export_rectangles`
        :meth:`.export_blocks`

        Parameters
        ----------
        path: :class:`str`
            Path to directory where rectangles were written.
        binary: :obj:`bool`
            If true, reads the files as binary, otherwise as text delimited.

        Returns
        -------
        :class:`numpy.ndarray`
        """
        def parse_rects(fname):
            rect_idx_and_bounds = [int(i) for i in re.findall(r'\d+', fname)]
            if len(rect_idx_and_bounds) != 5:
                raise ValueError(f'Invalid rectangle file name: {fname}')
            return rect_idx_and_bounds

        rect_files = [file['path'] for file in hl.utils.hadoop_ls(path) if not re.match(r'.*\.crc', file['path'])]
        rects = [parse_rects(os.path.basename(file_path)) for file_path in rect_files]

        n_rows = max(rects, key=lambda r: r[2])[2]
        n_cols = max(rects, key=lambda r: r[4])[4]

        nd = np.zeros(shape=(n_rows, n_cols))
        with with_local_temp_file() as f:
            uri = local_path_uri(f)
            for rect, file_path in zip(rects, rect_files):
                hl.utils.hadoop_copy(file_path, uri)
                if binary:
                    rect_data = np.reshape(np.fromfile(f), (rect[2] - rect[1], rect[4] - rect[3]))
                else:
                    rect_data = np.loadtxt(f, ndmin=2)
                nd[rect[1]:rect[2], rect[3]:rect[4]] = rect_data
        return nd

    @typecheck_method(compute_uv=bool,
                      complexity_bound=int)
    def svd(self, compute_uv=True, complexity_bound=8192):
        r"""Computes the reduced singular value decomposition.

        Examples
        --------

        >>> x = BlockMatrix.from_numpy(np.array([[-2.0, 0.0, 3.0],
        ...                                      [-1.0, 2.0, 4.0]]))
        >>> x.svd()
        (array([[-0.60219551, -0.79834865],
                [-0.79834865,  0.60219551]]),
         array([5.61784832, 1.56197958]),
         array([[ 0.35649586, -0.28421866, -0.89001711],
                [ 0.6366932 ,  0.77106707,  0.00879404]]))

        Notes
        -----
        This method leverages distributed matrix multiplication to compute
        reduced `singular value decomposition
        <https://en.wikipedia.org/wiki/Singular-value_decomposition>`__ (SVD)
        for matrices that would otherwise be too large to work with locally,
        provided that at least one dimension is less than or equal to 46300.

        Let :math:`X` be an :math:`n \times m` matrix and let
        :math:`r = \min(n, m)`. In particular, :math:`X` can have at most
        :math:`r` non-zero singular values. The reduced SVD of :math:`X`
        has the form

        .. math::

          X = U \Sigma V^T

        where

        - :math:`U` is an :math:`n \times r` matrix whose columns are
          (orthonormal) left singular vectors,

        - :math:`\Sigma` is an :math:`r \times r` diagonal matrix of non-negative
          singular values in descending order,

        - :math:`V^T` is an :math:`r \times m` matrix whose rows are
          (orthonormal) right singular vectors.

        If the singular values in :math:`\Sigma` are distinct, then the
        decomposition is unique up to multiplication of corresponding left and
        right singular vectors by -1. The computational complexity of SVD is
        roughly :math:`nmr`.

        We now describe the implementation in more detail.
        If :math:`\sqrt[3]{nmr}` is less than or equal to `complexity_bound`,
        then :math:`X` is localized to an ndarray on which
        :func:`scipy.linalg.svd` is called. In this case, all components are
        returned as ndarrays.

        If :math:`\sqrt[3]{nmr}` is greater than `complexity_bound`, then the
        reduced SVD is computed via the smaller gramian matrix of :math:`X`. For
        :math:`n > m`, the three stages are:

        1. Compute (and localize) the gramian matrix :math:`X^T X`,

        2. Compute the eigenvalues and right singular vectors via the
           symmetric eigendecomposition :math:`X^T X = V S V^T` with
           :func:`numpy.linalg.eigh` or :func:`scipy.linalg.eigh`,

        3. Compute the singular values as :math:`\Sigma = S^\frac{1}{2}` and the
           the left singular vectors as the block matrix
           :math:`U = X V \Sigma^{-1}`.

        In this case, since block matrix multiplication is lazy, it is efficient
        to subsequently slice :math:`U` (e.g. based on the singular values), or
        discard :math:`U` entirely.

        If :math:`n \leq m`, the three stages instead use the gramian
        :math:`X X^T = U S U^T` and return :math:`V^T` as the
        block matrix :math:`\Sigma^{-1} U^T X`.

        Warning
        -------
        Computing reduced SVD via the gramian presents an added wrinkle when
        :math:`X` is not full rank, as the block-matrix-side null-basis is not
        computable by the formula in the third stage. Furthermore, due to finite
        precision, the zero eigenvalues of :math:`X^T X` or :math:`X X^T` will
        only be approximately zero.

        If the rank is not known ahead, examining the relative sizes of the
        trailing singular values should reveal where the spectrum switches from
        non-zero to "zero" eigenvalues.  With 64-bit floating point, zero
        eigenvalues are typically about 1e-16 times the largest eigenvalue.
        The corresponding singular vectors should be sliced away **before** an
        action which realizes the block-matrix-side singular vectors.

        :meth:`svd` sets the singular values corresponding to negative
        eigenvalues to exactly ``0.0``.

        Warning
        -------
        The first and third stages invoke distributed matrix multiplication with
        parallelism bounded by the number of resulting blocks, whereas the
        second stage is executed on the leader (master) node.  For matrices of
        large minimum dimension, it may be preferable to run these stages
        separately.

        The performance of the second stage depends critically on the number of
        leader (master) cores and the NumPy / SciPy configuration, viewable with
        ``np.show_config()``. For Intel machines, we recommend installing the
        `MKL <https://anaconda.org/anaconda/mkl>`__ package for Anaconda.

        Consequently, the optimal value of `complexity_bound` is highly
        configuration-dependent.

        Parameters
        ----------
        compute_uv: :obj:`bool`
            If False, only compute the singular values (or eigenvalues).
        complexity_bound: :obj:`int`
            Maximum value of :math:`\sqrt[3]{nmr}` for which
            :func:`scipy.linalg.svd` is used.

        Returns
        -------
        u: :class:`numpy.ndarray` or :class:`BlockMatrix`
            Left singular vectors :math:`U`, as a block matrix if :math:`n > m` and
            :math:`\sqrt[3]{nmr}` exceeds `complexity_bound`.
            Only returned if `compute_uv` is True.
        s: :class:`numpy.ndarray`
            Singular values from :math:`\Sigma` in descending order.
        vt: :class:`numpy.ndarray` or :class:`BlockMatrix`
            Right singular vectors :math:`V^T``, as a block matrix if :math:`n \leq m` and
            :math:`\sqrt[3]{nmr}` exceeds `complexity_bound`.
            Only returned if `compute_uv` is True.
        """
        n, m = self.shape

        if n * m * min(n, m) <= complexity_bound ** 3:
            return _svd(self.to_numpy(), full_matrices=False, compute_uv=compute_uv, overwrite_a=True)
        else:
            return self._svd_gramian(compute_uv)

    @typecheck_method(compute_uv=bool)
    def _svd_gramian(self, compute_uv):
        x = self
        n, m = x.shape
        min_dim = min(n, m)
        if min_dim > 46300:  # limit due to localizing through Java array
            raise ValueError(f'svd: dimensions {n} and {m} both exceed 46300')

        left_gramian = n <= m
        a = ((x @ x.T if left_gramian else x.T @ x)
             .sparsify_triangle(lower=True, blocks_only=True)
             .to_numpy())

        if compute_uv:
            e, w = _eigh(a)
            for i in range(np.searchsorted(e, 0.0)):
                e[i] = 0

            # flip singular values to descending order
            s = np.flip(np.sqrt(e), axis=0)
            w = np.fliplr(w)

            if left_gramian:
                u = w
                vt = BlockMatrix.from_numpy((w / s).T) @ x
            else:
                u = x @ (w / s)
                vt = w.T
            return u, s, vt
        else:
            e = np.linalg.eigvalsh(a)
            for i in range(np.searchsorted(e, 0.0)):
                e[i] = 0

            return np.flip(np.sqrt(e), axis=0)


block_matrix_type.set(BlockMatrix)


def _is_scalar(x):
    return isinstance(x, float) or isinstance(x, int)


def _shape_after_broadcast(left, right):
    """
    Follows numpy's strategy of broadcasting through right-align shapes and
    compare corresponding dimensions. See:
    https://docs.scipy.org/doc/numpy-1.15.0/user/basics.broadcasting.html#general-broadcasting-rules
    """
    def join_dim(l_size, r_size):
        if not (l_size == r_size or l_size == 1 or r_size == 1):
            raise ValueError(f'Incompatible shapes for broadcasting: {left}, {right}')

        return max(l_size, r_size)

    def pad(arr, n):
        return [1 for _ in range(n)] + arr

    diff_len = len(left) - len(right)
    if diff_len < 0:
        left = pad(left, -diff_len)
    elif diff_len > 0:
        right = pad(right, diff_len)

    return [join_dim(lx, rx) for lx, rx in zip(left, right)]


@typecheck(x=oneof(numeric, np.ndarray), block_size=int)
def _to_bmir(x, block_size):
    if _is_scalar(x):
        return ValueToBlockMatrix(F64(x), [1, 1], block_size)
    else:
        data = list(_ndarray_as_float64(x).flat)
        return ValueToBlockMatrix(hl.literal(data)._ir, list(_ndarray_as_2d(x).shape), block_size)


def _broadcast_to_shape(bmir, result_shape):
    in_index_expr = _broadcast_index_expr(bmir.typ.shape, bmir.typ.is_row_vector)
    return BlockMatrixBroadcast(bmir, in_index_expr, result_shape, bmir.typ.block_size)


def _broadcast_index_expr(bmir_shape, is_row_vector):
    if len(bmir_shape) == 0:
        return []
    elif len(bmir_shape) == 1:
        return [1] if is_row_vector else [0]
    else:
        raise ValueError(f'Cannot broadcast shape: ${bmir_shape}')


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
    with with_local_temp_file() as path:
        uri = local_path_uri(path)
        nd.tofile(path)
        return Env.hail().utils.richUtils.RichArray.importFromDoubles(Env.spark_backend('_jarray_from_ndarray').fs._jfs, uri, nd.size)


def _ndarray_from_jarray(ja):
    with with_local_temp_file() as path:
        uri = local_path_uri(path)
        Env.hail().utils.richUtils.RichArray.exportToDoubles(Env.spark_backend('_ndarray_from_jarray').fs._jfs, uri, ja)
        return np.fromfile(path)


def _breeze_fromfile(uri, n_rows, n_cols):
    _check_entries_size(n_rows, n_cols)

    return Env.hail().utils.richUtils.RichDenseMatrixDouble.importFromDoubles(Env.spark_backend('_breeze_fromfile').fs._jfs, uri, n_rows, n_cols, True)


def _check_entries_size(n_rows, n_cols):
    n_entries = n_rows * n_cols
    if n_entries >= 1 << 31:
        raise ValueError(f'number of entries must be less than 2^31, found {n_entries}')


def _breeze_from_ndarray(nd):
    if any(i == 0 for i in nd.shape):
        raise ValueError(f'from_numpy: ndarray dimensions must be non-zero, found shape {nd.shape}')

    nd = _ndarray_as_2d(nd)
    nd = _ndarray_as_float64(nd)
    n_rows, n_cols = nd.shape

    with with_local_temp_file() as path:
        uri = local_path_uri(path)
        nd.tofile(path)
        return _breeze_fromfile(uri, n_rows, n_cols)


def _svd(a, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True):
    """
    SciPy supports two Lapack algorithms:
    DC: https://software.intel.com/en-us/mkl-developer-reference-fortran-gesdd
    GR: https://software.intel.com/en-us/mkl-developer-reference-fortran-gesvd
    DC (gesdd) is faster but uses O(elements) memory; lwork may overflow int32
    """
    try:
        return spla.svd(a, full_matrices=full_matrices, compute_uv=compute_uv, overwrite_a=overwrite_a,
                        check_finite=check_finite, lapack_driver='gesdd')
    except ValueError as e:
        if 'Too large work array required' in str(e):
            return spla.svd(a, full_matrices=full_matrices, compute_uv=compute_uv, overwrite_a=overwrite_a,
                            check_finite=check_finite, lapack_driver='gesvd')
        else:
            raise


def _eigh(a):
    """
    Only the lower triangle is used. Returns eigenvalues, eigenvectors.
    NumPy and SciPy apply different Lapack algorithms:
    NumPy uses DC: https://software.intel.com/en-us/mkl-developer-reference-fortran-syevd
    SciPy uses RRR: https://software.intel.com/en-us/mkl-developer-reference-fortran-syevr
    DC (syevd) is faster but uses O(elements) memory; lwork overflows int32 for dim_a > 32766
    """
    return np.linalg.eigh(a) if a.shape[0] <= 32766 else spla.eigh(a)
