from decorator import decorator
from hail.api2 import MatrixTable
from hail.utils.java import *
from hail.typecheck.check import *

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
@typecheck(dataset=MatrixTable, mapping=listof(strlike))
def reorder_columns(dataset, mapping):
    """Reorder samples.

    **Examples**

    Randomly shuffle order of samples:

    >>> import random
    >>> new_sample_order = vds.sample_ids[:]
    >>> random.shuffle(new_sample_order)
    >>> vds_reordered = vds.reorder_samples(new_sample_order)


    **Notes**

    This method requires unique sample ids. ``mapping`` must contain the same ids
    as :py:meth:`~hail.VariantDataset.sample_ids`. The order of the ids in ``mapping``
    determines the sample id order in the output dataset.


    :param mapping: New ordering of sample ids.
    :type mapping: list of str

    :return: Dataset with samples reordered.
    :rtype: :class:`.VariantDataset`
    """

    jvds = dataset._jvds.reorderSamples(mapping)
    return MatrixTable(jvds)
