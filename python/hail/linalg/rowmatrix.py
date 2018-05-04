from hail.utils.java import Env
from hail.typecheck import *
from hail.expr.expressions import expr_float64, matrix_table_source, check_entry_indexed

row_matrix_type = lazy()


class RowMatrix(object):
    """Hail's row-distributed matrix of :py:data:`.tfloat64` elements.

    .. include:: ../_templates/experimental.rst

    Notes
    -----
    Under the hood, row matrices are partitioned into group of rows.
    """

    def __init__(self, jrm):
        self._jrm = jrm

    @property
    def n_rows(self):
        """Number of rows.

        Returns
        -------
        :obj:`int`
        """
        return self._jrm.nRows()

    @property
    def n_cols(self):
        """Number of columns.

        Returns
        -------
        :obj:`int`
        """
        return self._jrm.nCols()

    @property
    def shape(self):
        """Shape of matrix.

        Returns
        -------
        (:obj:`int`, :obj:`int`)
           Number of rows and number of columns.
        """
        return self.n_rows, self.n_cols

    @classmethod
    @typecheck_method(entry_expr=expr_float64)
    def from_entry_expr(cls, entry_expr):
        """Creates a row matrix using a matrix table entry expression.

        Parameters
        ----------
        entry_expr: :class:`.Float64Expression`
            Entry expression for numeric matrix entries.

        Returns
        -------
        :class:`RowMatrix`
        """

        check_entry_indexed('write_from_entry_expr/entry_expr', entry_expr)

        mt = matrix_table_source('write_from_entry_expr/entry_expr', entry_expr)

        #  FIXME: remove once select_entries on a field is free
        if entry_expr in mt._fields_inverse:
            field = mt._fields_inverse[entry_expr]
            jrm = mt._jvds.toRowMatrix(field)
        else:
            field = Env.get_uid()
            jrm = mt.select_entries(**{field: entry_expr})._jvds.toRowMatrix(field)

        return cls(jrm)

    @classmethod
    @typecheck_method(path=str, partition_size=int)
    def read_block_matrix(cls, path, partition_size):
        """Creates a row matrix from a stored block matrix.

        Notes
        -----
        The number of partitions in the resulting row matrix will equal
        the ceiling of ``n_rows / partition_size``.

        Setting the partition size to an exact (rather than approximate)
        divisor or multiple of the block size will reduce superfluous shuffling
        of data.

        Warning
        -------
        The block matrix must be stored in row-major format, as results from
        :meth:`.BlockMatrix.write` with ``force_row_major=True`` and from
        :meth:`.BlockMatrix.write_from_entry_expr`. Otherwise,
        :meth:`read_block_matrix` will produce an error message.

        Parameters
        ----------
        path: :obj:`str`
            Path to block matrix.
        partition_size: :obj:`int`
            Number of rows to group per partition.
        Returns
        -------
        :class:`RowMatrix`
        """
        jrm = Env.hail().linalg.RowMatrix.readBlockMatrix(Env.hc()._jhc, path, partition_size)

        return cls(jrm)


row_matrix_type.set(RowMatrix)
