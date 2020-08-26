from hail import ir
from hail.expr import analyze
from hail.expr.expressions import expr_float64
from hail.table import Table
from hail.matrixtable import MatrixTable
from hail.methods.misc import require_biallelic, require_col_key_str
from hail.typecheck import typecheck, nullable, numeric


@typecheck(dataset=MatrixTable,
           maf=nullable(expr_float64),
           bounded=bool,
           min=nullable(numeric),
           max=nullable(numeric))
def identity_by_descent(dataset, maf=None, bounded=True, min=None, max=None) -> Table:
    """Compute matrix of identity-by-descent estimates.

    .. include:: ../_templates/req_tstring.rst

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------

    To calculate a full IBD matrix, using minor allele frequencies computed
    from the dataset itself:

    >>> hl.identity_by_descent(dataset)

    To calculate an IBD matrix containing only pairs of samples with
    ``PI_HAT`` in :math:`[0.2, 0.9]`, using minor allele frequencies stored in
    the row field `panel_maf`:

    >>> hl.identity_by_descent(dataset, maf=dataset['panel_maf'], min=0.2, max=0.9)

    Notes
    -----

    The dataset must have a column field named `s` which is a :class:`.StringExpression`
    and which uniquely identifies a column.

    The implementation is based on the IBD algorithm described in the `PLINK
    paper <http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1950838>`__.

    :func:`.identity_by_descent` requires the dataset to be biallelic and does
    not perform LD pruning. Linkage disequilibrium may bias the result so
    consider filtering variants first.

    The resulting :class:`.Table` entries have the type: *{ i: String,
    j: String, ibd: { Z0: Double, Z1: Double, Z2: Double, PI_HAT: Double },
    ibs0: Long, ibs1: Long, ibs2: Long }*. The key list is: `*i: String, j:
    String*`.

    Conceptually, the output is a symmetric, sample-by-sample matrix. The
    output table has the following form

    .. code-block:: text

        i		j	ibd.Z0	ibd.Z1	ibd.Z2	ibd.PI_HAT ibs0	ibs1	ibs2
        sample1	sample2	1.0000	0.0000	0.0000	0.0000 ...
        sample1	sample3	1.0000	0.0000	0.0000	0.0000 ...
        sample1	sample4	0.6807	0.0000	0.3193	0.3193 ...
        sample1	sample5	0.1966	0.0000	0.8034	0.8034 ...

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Variant-keyed and sample-keyed :class:`.MatrixTable` containing genotype information.
    maf : :class:`.Float64Expression`, optional
        Row-indexed expression for the minor allele frequency.
    bounded : :obj:`bool`
        Forces the estimations for `Z0``, ``Z1``, ``Z2``, and ``PI_HAT`` to take
        on biologically meaningful values (in the range [0,1]).
    min : :obj:`float` or :obj:`None`
        Sample pairs with a ``PI_HAT`` below this value will
        not be included in the output. Must be in :math:`[0,1]`.
    max : :obj:`float` or :obj:`None`
        Sample pairs with a ``PI_HAT`` above this value will
        not be included in the output. Must be in :math:`[0,1]`.

    Returns
    -------
    :class:`.Table`
    """

    require_col_key_str(dataset, 'identity_by_descent')

    if maf is not None:
        analyze('identity_by_descent/maf', maf, dataset._row_indices)
        dataset = dataset.select_rows(__maf=maf)
    else:
        dataset = dataset.select_rows()
    dataset = dataset.select_cols().select_globals().select_entries('GT')
    dataset = require_biallelic(dataset, 'ibd')

    return Table(ir.MatrixToTableApply(dataset._mir, {
        'name': 'IBD',
        'mafFieldName': '__maf' if maf is not None else None,
        'bounded': bounded,
        'min': min,
        'max': max,
    }))
