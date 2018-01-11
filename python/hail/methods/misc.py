from decorator import decorator
from hail.api2 import MatrixTable, Table
from hail.utils.java import Env, handle_py4j
from hail.typecheck.check import typecheck, strlike
from hail.expr.expression import *

@handle_py4j
@typecheck(table=Table, i=strlike, j=strlike, tie_breaker=nullable(strlike))
def maximal_independent_set(table, i, j, tie_breaker=None):
    """Compute a `maximal independent set
    <https://en.wikipedia.org/wiki/Maximal_independent_set>`__ of vertices in
    an undirected graph whose edges are given by this key table.

    Examples
    --------

    Prune individuals from a dataset until no close relationships remain with
    respect to a PC-Relate measure of kinship.

    >>> vds = hc.import_vcf("data/sample.vcf.bgz")
    >>> related_pairs = vds.pc_relate(2, 0.001).filter("kin > 0.125")
    >>> related_samples = related_pairs.query('i.flatMap(i => [i,j]).collectAsSet()')
    >>> related_samples_to_keep = related_pairs.maximal_independent_set("i", "j")
    >>> related_samples_to_remove = related_samples - set(related_samples_to_keep)
    >>> vds.filter_samples_list(list(related_samples_to_remove))

    Prune individuals from a dataset, prefering to keep cases over controls.

    >>> vds = hc.read("data/example.vds")
    >>> related_pairs = vds.pc_relate(2, 0.001).filter("kin > 0.125")
    >>> related_samples = related_pairs.query('i.flatMap(i => [i,j]).collectAsSet()')
    >>> related_samples_to_keep = (related_pairs
    ...   .key_by("i").join(vds.samples_table()).annotate('iAndCase = { id: i, isCase: sa.isCase }')
    ...   .select(['j', 'iAndCase'])
    ...   .key_by("j").join(vds.samples_table()).annotate('jAndCase = { id: j, isCase: sa.isCase }')
    ...   .select(['iAndCase', 'jAndCase'])
    ...   .maximal_independent_set("iAndCase", "jAndCase",
    ...     'if (l.isCase && !r.isCase) -1 else if (!l.isCase && r.isCase) 1 else 0'))
    >>> related_samples_to_remove = related_samples - {x.id for x in related_samples_to_keep}
    >>> vds.filter_samples_list(list(related_samples_to_remove))

    **Notes**

    The vertex set of the graph is implicitly all the values realized by
    ``i`` and ``j`` on the rows of this key table. Each row of the key table
    corresponds to an undirected edge between the vertices given by
    evaluating ``i`` and ``j`` on that row. An undirected edge may appear
    multiple times in the key table and will not affect the output. Vertices
    with self-edges are removed as they are not independent of themselves.

    The expressions for ``i`` and ``j`` must have the same type.

    This method implements a greedy algorithm which iteratively removes a
    vertex of highest degree until the graph contains no edges.

    ``tie_breaker`` is a Hail expression that defines an ordering on
    nodes. It has two values in scope, ``l`` and ``r``, that refer the two
    nodes being compared. A pair of nodes can be ordered in one of three
    ways, and ``tie_breaker`` must encode the relationship as follows:

     - if ``l < r`` then ``tie_breaker`` evaluates to some negative integer
     - if ``l == r`` then ``tie_breaker`` evaluates to 0
     - if ``l > r`` then ``tie_breaker`` evaluates to some positive integer

    For example, the usual ordering on the integers is defined by: ``l - r``.

    When multiple nodes have the same degree, this algorithm will order the
    nodes according to ``tie_breaker`` and remove the *largest* node.

    :param str i: expression to compute one endpoint.
    :param str j: expression to compute another endpoint.
    :param tie_breaker: Expression used to order nodes with equal degree.

    :return: a list of vertices in a maximal independent set.
    :rtype: list of elements with the same type as ``i`` and ``j``

    """

    return jarray_to_list(self._jkt.maximalIndependentSet(i, j, joption(tie_breaker)))

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

