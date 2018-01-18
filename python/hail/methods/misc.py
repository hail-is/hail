from decorator import decorator
from hail.api2 import MatrixTable
from hail.utils.java import Env, handle_py4j
from hail.typecheck.check import typecheck, strlike
from hail.expr.expression import *

@handle_py4j
@typecheck(dataset=MatrixTable, method=strlike)
def require_biallelic(dataset, method):
    from hail.expr.types import TVariant
    if not isinstance(dataset.rowkey_schema, TVariant):
        raise TypeError("Method '{}' requires the row key to be of type 'TVariant', found '{}'".format(
            method, dataset.rowkey_schema))
    dataset = MatrixTable(Env.hail().methods.VerifyBiallelic.apply(dataset._jvds, method))
    return dataset

@handle_py4j
@typecheck(dataset=MatrixTable)
def rename_duplicates(dataset):
    """Rename duplicate column keys.

    .. include:: ../_templates/req_tstring.rst

    Examples
    --------

    >>> renamed = methods.rename_duplicates(dataset).cols_table()
    >>> duplicate_samples = (renamed.filter(renamed.s != renamed.originalID)
    ...                             .select('originalID')
    ...                             .collect())

    Notes
    -----

    This method produces a dataset with unique column keys by appending a unique
    suffix ``_N`` to duplicate keys. For example, if the column key "NA12878"
    appears three times in the dataset, the first will be left as "NA12878", the
    second will be renamed "NA12878_1", and the third will be "NA12878_2". The
    original column key is stored in the column field `originalID`.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.

    Returns
    -------
    :class:`.MatrixTable`
        Dataset with duplicate column keys renamed.
    """

    return MatrixTable(dataset._jvds.renameDuplicates())

@handle_py4j
def to_matrix_table(table, row_key, col_key, **entry_exprs):

    all_exprs = []
    row_key = to_expr(row_key)
    all_exprs.append(row_key)
    analyze('to_matrix_table/row_key', row_key, table._row_indices)
    col_key = to_expr(col_key)
    all_exprs.append(col_key)
    analyze('to_matrix_table/col_key', col_key, table._row_indices)

    exprs = []

    entry_exprs = {k: to_expr(v) for k, v in entry_exprs.items()}
    for k, e in entry_exprs.items():
        all_exprs.append(e)
        analyze('to_matrix_table/entry_exprs/{}'.format(k), e, table._row_indices)
        exprs.append('`{k}` = {v}'.format(k=k, v=e._ast.to_hql()))

    base, cleanup = table._process_joins(*all_exprs)

    return MatrixTable(base._jt.toMatrixTable(row_key._ast.to_hql(),
                                              col_key._ast.to_hql(),
                                              ",\n".join(exprs),
                                              joption(None)))

