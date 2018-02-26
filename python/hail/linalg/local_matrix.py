from hail.utils.java import Env, scala_object, jarray, numpy_from_breeze, FatalError
from hail.typecheck import *
import hail.linalg
import numpy as np

local_matrix_type = lazy()


class LocalMatrix(object):
    """Hail's non-distributed matrix of :py:data:`.tfloat64` elements.

    Notes
    -----

    The binary operators ``+``, ``-``, ``*``, and ``/`` act pointwise and
    follow NumPy ``ndarray``
    `broadcast rules <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`__.
    Each operand can be a local matrix, an integer, or a float.
    Use ``**`` for pointwise exponentiation. Use ``@`` to multiply two local
    matrices using matrix multiplication.

    Local matrices must have fewer than :math:`2^{31}` elements (16GB of data)
    and are further limited by available memory on the master node.
    """

    def __init__(self, jlm):
        self._jlm = jlm

    @classmethod
    @typecheck_method(path=str)
    def read(cls, path):
        """Read a local matrix.

        Parameters
        ----------

        path: :obj:`str`
            Path to input file.

        Returns
        -------
        :class:`.LocalMatrix`
        """

        hc = Env.hc()
        return cls(Env.hail().linalg.LocalMatrix.read(hc._jhc, path))

    @staticmethod
    @typecheck_method(num_rows=int,
                      num_cols=int,
                      data=listof(float),
                      is_transpose=bool)
    def _create(num_rows, num_cols, data, is_transpose):
        return LocalMatrix(scala_object(Env.hail().linalg, 'LocalMatrix').apply(
            num_rows, num_cols, jarray(Env.jvm().double, data), is_transpose))

    @staticmethod
    def from_ndarray(ndarray):
        """Create a local matrix from a NumPy ``ndarray``.

        Notes
        -----
        This method is not efficient for large matrices.

        Parameters
        ----------

        ndarray: NumPy ``ndarray``
            ``ndarray`` with two dimensions, each of non-zero size.

        Returns
        -------
        :class:`.LocalMatrix`
        """

        # convert to float64 to ensure that jarray method below works
        ndarray = ndarray.astype(np.float64)
        if ndarray.ndim != 2:
            raise FatalError("from_ndarray: ndarray must have two axes, found shape {}".format(ndarray.shape))
        num_rows, num_cols = ndarray.shape[0:2]
        if num_rows==0 or num_cols==0:
            raise FatalError("from_ndarray: ndarray dimensions must be non-zero, found shape {}".format(ndarray.shape))

        # np.ravel() exports row major by default
        data = jarray(Env.jvm().double, np.ravel(ndarray))
        # breeze is column major
        is_transpose = True

        return LocalMatrix(scala_object(Env.hail().linalg, 'LocalMatrix').apply(
            num_rows, num_cols, jarray(Env.jvm().double, data), is_transpose))

    @staticmethod
    @typecheck_method(num_rows=int,
                      num_cols=int,
                      seed=int,
                      uniform=bool)
    def random(num_rows, num_cols, seed=0, uniform=True):
        """Create a local matrix with uniform or normal random entries.

        Parameters
        ----------

        num_rows: :obj:`int`
            Number of rows.

        num_cols: :obj:`int`
            Number of columns.

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

        return LocalMatrix(Env.hail().linalg.LocalMatrix.random(num_rows, num_cols, seed, uniform))

    @property
    def num_rows(self):
        """Number of rows.

        Returns
        -------
        :obj:`int`
        """

        return self._jlm.nRows()

    @property
    def num_cols(self):
        """Number of columns.

        Returns
        -------
        :obj:`int`
        """
        return self._jlm.nCols()

    @property
    def shape(self):
        """Number of rows and number of columns.

        Returns
        -------
        (:obj:`int`, :obj:`int`)
        """
        return self.num_rows, self.num_cols

    @typecheck_method(path=str)
    def write(self, path):
        """Write the local matrix.

        Parameters
        ----------

        path: :obj:`str`
            Path for output file.
        """

        self._jlm.write(path)

    def to_ndarray(self):
        """Create a NumPy ``ndarray`` from the local matrix.

        Returns
        -------
        NumPy ``ndarray``
            ``ndarray`` with two dimensions, each of non-zero size.
        """

        return numpy_from_breeze(self._jlm.m())

    @typecheck_method(block_size=nullable(int))
    def to_block_matrix(self, block_size=None):
        """Create a block matrix by distributing a local matrix.

        Notes
        -----
        This method creates a blocked copy of the local matrix in memory
        so may be unreliable when the local matrix is very large.
        In this case, fall back to the safer approach of writing with
        :meth:`LocalMatrix.write_block_matrix` and then reading with
        :meth:`BlockMatrix.read`.

        Parameters
        ----------

        block_size: :obj:`str`
            Block size of the resulting block matrix, optional.
            Default given by :meth:`~BlockMatrix.default_block_size`.

        Returns
        -------
        :class:`.BlockMatrix`
        """

        if not block_size:
            block_size = hail.linalg.BlockMatrix.default_block_size()

        hc = Env.hc()
        return hail.linalg.BlockMatrix(self._jlm.toBlockMatrix(hc._jhc, block_size))

    @typecheck_method(path=str,
                      block_size=nullable(int))
    def write_block_matrix(self, path, block_size=None, force_row_major=False):
        """Write the local matrix in block matrix format.

        Parameters
        ----------

        path: :obj:`str`
            Path for output file.

        block_size: :obj:`str`
            Block size, optional.
            Default given by :meth:`~BlockMatrix.default_block_size`.

        force_row_major: :obj:`bool`
            If ``True``, transform blocks in column-major format
            to row-major format before writing.
            If ``False``, write blocks in their current format.
        """

        if not block_size:
            block_size = hail.linalg.BlockMatrix.default_block_size()

        hc = Env.hc()
        self._jlm.writeBlockMatrix(hc._jhc, path, block_size, force_row_major)

    # add elements and slicing

    @typecheck_method(rows_to_keep=listof(int))
    def filter_rows(self, rows_to_keep):
        """Filter matrix rows.

        Parameters
        ----------

        rows_to_keep: :obj:`list` of :obj:`int`
            Indices of rows to keep. Must be non-empty and increasing.

        Returns
        -------
        :class:`.LocalMatrix`
        """

        return LocalMatrix(self._jlm.filterRows(jarray(Env.jvm().int, rows_to_keep)))

    @typecheck_method(cols_to_keep=listof(int))
    def filter_cols(self, cols_to_keep):
        """Filter matrix columns.

        Parameters
        ----------

        cols_to_keep: :obj:`list` of :obj:`int`
            Indices of columns to keep. Must be non-empty and increasing.

        Returns
        -------
        :class:`.LocalMatrix`
        """

        return LocalMatrix(self._jlm.filterCols(jarray(Env.jvm().int, cols_to_keep)))

    def T(self):
        """Matrix transpose.

        Returns
        -------
        :class:`.LocalMatrix`
        """

        return LocalMatrix(self._jlm.t())

    def diagonal(self):
        """Returns the diagonal as a local matrix with one column.

        Returns
        -------
        :class:`.LocalMatrix`
        """

        return LocalMatrix(self._jlm.diagonal())

    def __pos__(self):
        return self

    def __neg__(self):
        """Negation: -A.

        Returns
        -------
        :class:`.LocalMatrix`
        """

        return LocalMatrix(self._jlm.unary_-())

    @typecheck_method(b=oneof(numeric, local_matrix_type))
    def __add__(self, b):
        """Addition with broadcast: A + B.

        Parameters
        ----------

        b: :class:`LocalMatrix` or :obj:`int` or :obj:`float`

        Returns
        -------
        :class:`.LocalMatrix`
        """

        if isinstance(b, int) or isinstance(b, float):
            return LocalMatrix(self._jlm.add(float(b)))
        else:
            return LocalMatrix(self._jlm.add(b._jlm))

    @typecheck_method(b=numeric)
    def __radd__(self, b):
        return self + b

    @typecheck_method(b=oneof(numeric, local_matrix_type))
    def __sub__(self, b):
        """Subtraction with broadcast: A - B.

        Parameters
        ----------

        b: :class:`LocalMatrix` or :obj:`int` or :obj:`float`

        Returns
        -------
        :class:`.LocalMatrix`
        """

        if isinstance(b, int) or isinstance(b, float):
            return LocalMatrix(self._jlm.subtract(float(b)))
        else:
            return LocalMatrix(self._jlm.subtract(b._jlm))

    @typecheck_method(b=numeric)
    def __rsub__(self, b):
        return LocalMatrix(self._jlm.rsubtract(float(b)))

    @typecheck_method(b=oneof(numeric, local_matrix_type))
    def __mul__(self, b):
        """Pointwise multiplication with broadcast: A * B.

        Parameters
        ----------

        b: :class:`LocalMatrix` or :obj:`int` or :obj:`float`

        Returns
        -------
        :class:`.LocalMatrix`
        """

        if isinstance(b, int) or isinstance(b, float):
            return LocalMatrix(self._jlm.multiply(float(b)))
        else:
            return LocalMatrix(self._jlm.multiply(b._jlm))

    @typecheck_method(b=numeric)
    def __rmul__(self, b):
        return self * b

    @typecheck_method(b=oneof(numeric, local_matrix_type))
    def __truediv__(self, b):
        """Pointwise division with broadcast: A / B.

        Parameters
        ----------

        b: :class:`LocalMatrix` or :obj:`int` or :obj:`float`

        Returns
        -------
        :class:`.LocalMatrix`
        """

        if isinstance(b, int) or isinstance(b, float):
            return LocalMatrix(self._jlm.divide(float(b)))
        else:
            return LocalMatrix(self._jlm.divide(b._jlm))

    @typecheck_method(b=numeric)
    def __rtruediv__(self, b):
        return LocalMatrix(self._jlm.rdivide(float(b)))

    def sqrt(self):
        """Pointwise squareroot.

        Returns
        -------
        :class:`.LocalMatrix`
        """

        return LocalMatrix(self._jlm.sqrt())

    @typecheck_method(exponent=numeric)
    def __pow__(self, x):
        """Pointwise exponentiation: A ** x.

        Parameters
        ----------
        x: :obj:`int` or :obj:`float`
            Exponent.

        Returns
        -------
        :class:`.LocalMatrix`
        """

        return LocalMatrix(self._jlm.pow(float(x)))

    @typecheck_method(b=local_matrix_type)
    def __matmul__(self, b):
        """Matrix multiplication: A @ B.

        Parameters
        ----------

        b: :class:`LocalMatrix`

        Returns
        -------
        :class:`.LocalMatrix`
        """

        if (self.num_cols != b.num_rows):
            raise FatalError('matmul: Invalid shapes for matrix multiplication: {}, {}'.format(self.shape, b.shape))

        return LocalMatrix(self._jlm.matrixMultiply(b._jlm))

    def inv(self):
        """Matrix inverse.

        Returns
        -------
        :class:`.LocalMatrix`
        """

        if (self.num_rows != self.num_cols):
            raise FatalError('inv: Matrix must be square, found shape {}'.format(self.shape))

        return LocalMatrix(self._jlm.inverse())

    @typecheck_method(b=local_matrix_type)
    def solve(self, b):
        """Solve AX=B for X.

        Notes
        -----
        A and B must have the same number of rows.
        Returns X, a matrix with same shape as B.

        Parameters
        ----------

        b: :class:`LocalMatrix`

        Returns
        -------
        :class:`.LocalMatrix`
        """

        if (self.num_rows != self.num_cols):
            raise FatalError('solve: Matrices must have the same number of rows, found shapes {}, {}'.format(self.shape, b.shape))

        return LocalMatrix(self._jlm.solve(b._jlm))

    def eigh(self):
        """Eigendecomposition of symmetric matrix.

        Notes
        -----

        The matrix A is assumed to be symmetric; only the lower triangle is used.

        The `eigendecomposition <https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Real_symmetric_matrices>`__
        of A has the form A = USU.T where U has
        orthonormal columns (eigenvectors) and S is diagonal
        (eigenvalues in increasing order)

        This method returns (diagonal(S), U). To avoid computing eigenvectors when
        only the eigenvalues are needed, use :meth:`LocalMatrix.eigvalsh`.

        Returns
        -------
        (:class:`.LocalMatrix`, :class:`.LocalMatrix`)
            Eigenvalues and eigenvectors.
        """

        if (self.num_rows != self.num_cols):
            raise FatalError('eigh: Matrix must be square, found shape {}'.format(self.shape))

        res = self._jlm.eigh()
        return LocalMatrix(res._1()), LocalMatrix(res._2())

    def eigvalsh(self):
        """Eigenvalues of symmetric matrix.

        Notes
        -----

        A is assumed to be symmetric; only the lower triangle is used.
        Eigenvalues are returned in increasing order as a matrix with
        a single column.

        Returns
        -------
        :class:`.LocalMatrix`
            Eigenvalues.
        """

        if (self.num_rows != self.num_cols):
            raise FatalError('eigvalsh: Matrix must be square, found shape {}'.format(self.shape))

        return LocalMatrix(self._jlm.eigvalsh())

    @typecheck_method(reduced=bool)
    def svd(self, reduced=False):
        """Singular-value decomposition.

        Notes
        -----

        The `singular-value decomposition <https://en.wikipedia.org/wiki/Singular-value_decomposition>`__
        of A has the form A = USV.T
        where U has orthonormal columns (left singular vectors),
        S is diagonal (singular values in increasing order),
        and V has orthonormal columns (right singular vectors).

        Suppose A has shape (n, m) and let k = min(n, m). This methods returns (U, diagonal(S), V^T).

        If ``reduced=False``, the shapes are (n, n), (k, 1), and (m, m).

        If ``reduced=True``, the shapes are (n, k),  (k, 1), and (k, m).

        Parameters
        ----------

        reduced: :obj:`bool`
            If ``True``, return the reduced SVD.

        Returns
        -------
        (:class:`.LocalMatrix`, :class:`.LocalMatrix`, :class:`.LocalMatrix`)
            Left singular vectors as columns, singular values as one column,
            and right singular vectors as rows.
        """

        res = self._jlm.svd(reduced)
        return LocalMatrix(res._1()), LocalMatrix(res._2()), LocalMatrix(res._3())

    @typecheck_method(reduced=bool)
    def qr(self, reduced=False):
        """QR decomposition.

        Notes
        -----

        The `QR decomposition <https://en.wikipedia.org/wiki/QR_decomposition>`__
        of A has the form A = QR
        where Q has orthonormal columns and R is upper triangular.

        Suppose A has shape (n, m) and let k = min(n, m).
        This methods returns (Q, R).

        If ``reduced=False``, the shapes are (n, m) and (m ,m).

        If ``reduced=True``, the shapes are (n, k) and (k, m).

        Parameters
        ----------

        reduced: :obj:`bool`
            If ``True``, return the reduced QR decomposition.

        Returns
        -------
        (:class:`.LocalMatrix`, :class:`.LocalMatrix`)
            (Q, R)
        """

        res = self._jlm.qr(reduced)
        return LocalMatrix(res._1()), LocalMatrix(res._2())

    def cholesky(self):
        """Cholesky decomposition of symmetric, positive-definite matrix.

        Notes
        -----

        The matrix A is assumed to be symmetric and positive-definite;
        only the lower triangle is used.

        The `Cholesky decomposition <https://en.wikipedia.org/wiki/Cholesky_decomposition>`__
        of A has the form A = LL.T where L is lower triangular
        with positive diagonal entries.

        Returns
        -------
        :class:`.LocalMatrix`
            Lower triangular Cholesky factor.
        """

        if (self.num_rows != self.num_cols):
            raise FatalError('cholesky: Matrix must be square, found shape {}'.format(self.shape))

        return LocalMatrix(self._jlm.cholesky())


local_matrix_type.set(LocalMatrix)
