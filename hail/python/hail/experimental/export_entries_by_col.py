import hail as hl
from hail.matrixtable import MatrixTable
from hail.typecheck import typecheck


@typecheck(
    mt=MatrixTable, path=str, batch_size=int, bgzip=bool, header_json_in_file=bool, use_string_key_as_file_name=bool
)
def export_entries_by_col(
    mt: MatrixTable,
    path: str,
    batch_size: int = 256,
    bgzip: bool = True,
    header_json_in_file: bool = True,
    use_string_key_as_file_name: bool = False,
):
    """Export entries of the `mt` by column as separate text files.

    Examples
    --------
    >>> range_mt = hl.utils.range_matrix_table(10, 10)
    >>> range_mt = range_mt.annotate_entries(x = hl.rand_unif(0, 1))
    >>> hl.experimental.export_entries_by_col(range_mt, 'output/cols_files')

    Notes
    -----
    This function writes a directory with one file per column in `mt`. The
    files contain one tab-separated field (with header) for each row field
    and entry field in `mt`. The column fields of `mt` are written as JSON
    in the first line of each file, prefixed with a ``#``.

    The above will produce a directory at ``output/cols_files`` with the
    following files:

    .. code-block:: text

        $ ls -l output/cols_files
        total 80
        -rw-r--r--  1 hail-dev  wheel  712 Jan 25 17:19 index.tsv
        -rw-r--r--  1 hail-dev  wheel  712 Jan 25 17:19 part-00.tsv.bgz
        -rw-r--r--  1 hail-dev  wheel  712 Jan 25 17:19 part-01.tsv.bgz
        -rw-r--r--  1 hail-dev  wheel  712 Jan 25 17:19 part-02.tsv.bgz
        -rw-r--r--  1 hail-dev  wheel  712 Jan 25 17:19 part-03.tsv.bgz
        -rw-r--r--  1 hail-dev  wheel  712 Jan 25 17:19 part-04.tsv.bgz
        -rw-r--r--  1 hail-dev  wheel  712 Jan 25 17:19 part-05.tsv.bgz
        -rw-r--r--  1 hail-dev  wheel  712 Jan 25 17:19 part-06.tsv.bgz
        -rw-r--r--  1 hail-dev  wheel  712 Jan 25 17:19 part-07.tsv.bgz
        -rw-r--r--  1 hail-dev  wheel  712 Jan 25 17:19 part-08.tsv.bgz
        -rw-r--r--  1 hail-dev  wheel  712 Jan 25 17:19 part-09.tsv.bgz

        $ zcat output/cols_files/part-00.tsv.bgz
        #{"col_idx":0}
        row_idx  x
        0        6.2501e-02
        1        7.0083e-01
        2        3.6452e-01
        3        4.4170e-01
        4        7.9177e-02
        5        6.2392e-01
        6        5.9920e-01
        7        9.7540e-01
        8        8.4848e-01
        9        3.7423e-01

    Due to overhead and file system limits related to having large numbers
    of open files, this function will iteratively export groups of columns.
    The `batch_size` parameter can control the size of these groups.

    Parameters
    ----------
    mt : :class:`.MatrixTable`
    path : :obj:`int`
        Path (directory to write to.
    batch_size : :obj:`int`
        Number of columns to write per iteration.
    bgzip : :obj:`bool`
        BGZip output files.
    header_json_in_file : :obj:`bool`
        Include JSON header in each component file (if False, only written to index.tsv)
    """
    if use_string_key_as_file_name and not (len(mt.col_key) == 1 and mt.col_key[0].dtype == hl.tstr):
        raise ValueError(
            f'parameter "use_string_key_as_file_name" requires a single string column key, found {list(mt.col_key.dtype.values())}'
        )
    hl.utils.java.Env.backend().execute(
        hl.ir.MatrixToValueApply(
            mt._mir,
            {
                'name': 'MatrixExportEntriesByCol',
                'parallelism': batch_size,
                'path': path,
                'bgzip': bgzip,
                'headerJsonInFile': header_json_in_file,
                'useStringKeyAsFileName': use_string_key_as_file_name,
            },
        )
    )
