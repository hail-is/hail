import hail as hl
from hail.utils.java import Env, joption
from hail.typecheck import *
from hail.expr.expressions import expr_float64, matrix_table_source, check_entry_indexed

row_matrix_type = lazy()


class RowMatrix(object):
    """Hail's row-distributed matrix of :py:data:`.tfloat64` elements.

    .. include:: ../_templates/experimental.rst

    Notes
    -----
    Row matrices must have fewer than :math:`2^{31}` columns.

    Under the hood, row matrices are partitioned into groups of rows.
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

        Examples
        --------

        >>> mt = hl.balding_nichols_model(3, 25, 50)
        >>> bm = RowMatrix.from_entry_expr(mt.GT.n_alt_alleles())

        Parameters
        ----------
        entry_expr: :class:`.Float64Expression`
            Entry expression for numeric matrix entries.
            All values must be non-missing.

        Returns
        -------
        :class:`RowMatrix`
        """

        check_entry_indexed('RowMatrix.from_entry_expr/entry_expr', entry_expr)

        mt = matrix_table_source('RowMatrix.from_entry_expr/entry_expr', entry_expr)

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
    def read_from_block_matrix(cls, path, partition_size):
        """Creates a row matrix from a stored block matrix.

        Examples
        --------
        >>> rm = RowMatrix.read_from_block_matrix('output/bm',
        ...                                       partition_size=2)

        Notes
        -----
        The number of partitions in the resulting row matrix equals
        the ceiling of ``n_rows / partition_size``.
        See :meth:`BlockMatrix.to_row_matrix` for more info.

        Warning
        -------
        The block matrix must be stored in row-major format, as results from
        :meth:`.BlockMatrix.write` with ``force_row_major=True`` and from
        :meth:`.BlockMatrix.write_from_entry_expr`. Otherwise,
        :meth:`read_from_block_matrix` will produce an error message.

        Parameters
        ----------
        path: :obj:`str`
            Path to block matrix on disk.
        partition_size: :obj:`int`
            Number of rows to group per partition.

        Returns
        -------
        :class:`RowMatrix`
        """
        jrm = Env.hail().linalg.RowMatrix.readBlockMatrix(Env.hc()._jhc, path, partition_size)

        return cls(jrm)

    @typecheck_method(path=str,
                      delimiter=str,
                      header=nullable(str),
                      add_index=bool,
                      parallel=nullable(enumeration('separate_header', 'header_per_shard')),
                      entries=enumeration('full', 'lower', 'strict_lower', 'upper', 'strict_upper'))
    def export(self, path, delimiter='\t', header=None, add_index=False, parallel=None, entries='full'):
        """Exports row matrix as a delimited text file.

        Examples
        --------
        Consider the following row matrix.

        >>> import numpy as np
        >>> from hail.linalg import BlockMatrix
        >>>
        >>> nd = np.array([[1.0, 0.8, 0.7],
        ...                [0.8, 1.0 ,0.3],
        ...                [0.7, 0.3, 1.0]])
        >>> bm = BlockMatrix.from_numpy(nd)
        >>> rm = bm.to_row_matrix(partition_size=2)

        Export the full matrix as a file with tab-separated values:

        >>> rm.export('output/row_matrix.tsv')

        Export the upper-triangle of the matrix as a file of
        comma-separated values.

        >>> rm.export('output/row_matrix.csv',
        ...           delimiter=',',
        ...           entries='upper')

        Export the full matrix with row indices in parallel as a folder of
        files, each with a header line for columns ``idx``, ``A``, ``B``,
        and ``C``.

        >>> rm.export('output/row_matrix',
        ...           header='\t'.join(['idx', 'A', 'B', 'C']),
        ...           add_index=True,
        ...           parallel='header_per_shard')

        Since ``rm`` has two partitions, this produces two file:

        .. code-block:: text

            idx A   B   C
            0   1.0 0.8 0.7
            1   0.8 1.0 0.3

        .. code-block:: text

            idx A   B   C
            2   0.7 0.3 1.0

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

        Parameters
        ----------
        path: :obj:`str`
            Path for export.
        delimiter: :obj:`str`
            Column delimiter.
        header: :obj:`str`, optional
            If provided, `header` is prepended before the first row of data.
        add_index: :obj:`bool`
            If ``True``, add an initial column with the absolute row index.
        parallel: :obj:`str`, optional
            If ``'header_per_shard'``, create a folder with one file per
            partition, each with a header if provided.
            If ``'separate_header'``, create a folder with one file per
            partition without a header; write the header, if provided, in
            a separate file.
            If ``None``, serially concatenate the header and all partitions
            into one file; export will be slower.
            If `header` is ``None`` then ``'header_per_shard'`` and
            ``'separate_header'`` are equivalent.
        entries: :obj:`str
            Describes which entries to export. One of:
            ``'full'``, ``'lower'``, ``'strict_lower'``, ``'upper'``, ``'strict_upper'``.
        """
        export_type = Env.hail().utils.ExportType.getExportType(parallel)

        if entries == 'full':
            self._jrm.export(path, delimiter, joption(header), add_index, export_type)
        elif entries == 'lower':
            self._jrm.exportLowerTriangle(path, delimiter, joption(header), add_index, export_type)
        elif entries == 'strict_lower':
            self._jrm.exportStrictLowerTriangle(path, delimiter, joption(header), add_index, export_type)
        elif entries == 'upper':
            self._jrm.exportUpperTriangle(path, delimiter, joption(header), add_index, export_type)
        else:
            assert entries == 'strict_upper'
            self._jrm.exportStrictUpperTriangle(path, delimiter, joption(header), add_index, export_type)


row_matrix_type.set(RowMatrix)
