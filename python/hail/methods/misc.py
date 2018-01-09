from decorator import decorator
from hail.api2 import MatrixTable
from hail.utils.java import Env, handle_py4j
from hail.typecheck.check import typecheck

@decorator
def require_biallelic(f, dataset, *args, **kwargs):
    from hail.expr.types import TVariant
    if not isinstance(dataset.rowkey_schema, TVariant):
        raise TypeError("Method '{}' requires the row key to be of type 'TVariant', found '{}'".format(
            f.__name__, dataset.rowkey_schema))
    dataset = MatrixTable(Env.hail().methods.VerifyBiallelic.apply(dataset._jvds, f.__name__))
    return f(dataset, *args, **kwargs)

@handle_py4j
@typecheck(dataset=MatrixTable)
def rename_duplicates(dataset):
    """Rename duplicate column keys.

    .. include:: ../_templates/req_tstring.rst

    Examples
    --------

    >>> renamed = methods.rename_duplicates(dataset).cols_table()
    >>> renamed = renamed.filter(renamed.s != renamed.originalID)
    >>> duplicate_samples = renamed.select(renamed.originalID).collect()

    Notes
    -----

    This method produces a dataset with unique column keys by appending a unique
    suffix ``_N`` to duplicate keys. For example, if the column key "NA12878"
    appears three times in the dataset, the first will be left as "NA12878", the
    second will be renamed "NA12878_1", and the third will be "NA12878_2". The
    original column key is stored in the column field `originalID`.

    Parameters
    ----------
    dataset : :class:`MatrixTable`
        Dataset.

    Returns
    -------
    :class:`MatrixTable`
        Dataset with duplicate column keys renamed.
    """

    return MatrixTable(dataset._jvds.renameDuplicates())
