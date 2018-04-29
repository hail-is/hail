from hail.utils.java import Env, joption
from hail.typecheck import *
from hail.expr.expressions import expr_float64, matrix_table_source, check_entry_indexed

row_matrix_type = lazy()


class RowMatrix(object):
    """Hail's row-distributed matrix of :py:data:`.tfloat64` elements.

    .. include:: ../_templates/experimental.rst

    Notes
    -----
    Row matrices must have fewer than :math:`2^31` columns.

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

        >>> import numpy as np
        >>> from hail.linalg import BlockMatrix
        >>> from hail.linalg import RowMatrix
        >>> nd = np.array([[1.0, 0.8, 0.7],
        ...                [0.8, 1.0 ,0.3],
        ...                [0.7, 0.3, 1.0]])
        >>> BlockMatrix.from_numpy(nd).write('output/bm', force_row_major=True)
        >>> rm = RowMatrix.read_block_matrix('output/bm', partition_size=2)

        Export the full matrix as a file with tab-separated values:

        >>> rm.export('output/row_matrix.tsv')

        Export the upper-triangle of the matrix as a file of
        comma-separated values.

        >>> rm.export('output/row_matrix.csv',
        ...           delimiter=',',
        ...           entries='upper')

        Export the full matrix in parallel as a folder of files,
        each with a header line for columns ``index``, ``A``, ``B``,
        and ``C``. Every value line is tab-separated and prepended
        by its absolute row index:

        >>> rm.export('output/row_matrix',
        ...           header='\t'.join(['index', 'A', 'B', 'C']),
        ...           add_index=True,
        ...           parallel='header_per_shard')

        Notes
        -----
        Here is the full export with header and row index as one file:

        .. code-block:: text

            index,A,B,C
            0   1.0 0.8 0.7
            1   0.8 1.0 0.3
            2   0.7 0.3 1.0

        The five `entries` options are illustrated below.

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

        If `header` is ``None``, the `parallel` options ``'header_per_shard'`` and
        ``'separate_header'`` are equivalent.

        Parameters
        ----------
        path: :obj:`str`
            Path for export.
        delimiter: :obj:`str`
            Column delimiter.
        header: :obj:`str`, optional
            If provided, header string is prepended before the first row of data.
        add_index: :obj:`bool`
            If true, add an initial column with the row index.
        parallel: :obj:`str`, optional
            If ``'header_per_shard'``, create a folder with one file per
            partition, each with a header if provided.
            If ``'separate_header'``, create a folder with one file per
            partition without a header; write the header, if provided, in
            a separate file.
            If ``None``, serially concatenate the header and all partitions
            into one file; export will be slower.
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
