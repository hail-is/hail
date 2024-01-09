import hail as hl
from hail.backend.spark_backend import SparkBackend
from hail.expr import analyze
from hail.expr.expressions import expr_float64
import hail.ir as ir
from hail.table import Table
from hail.matrixtable import MatrixTable
from hail.methods.misc import require_biallelic, require_col_key_str
from hail.typecheck import typecheck, nullable, numeric
from hail.linalg import BlockMatrix
from hail.utils.java import Env


@typecheck(dataset=MatrixTable, maf=nullable(expr_float64), bounded=bool, min=nullable(numeric), max=nullable(numeric))
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
        Forces the estimations for ``Z0``, ``Z1``, ``Z2``, and ``PI_HAT`` to take
        on biologically meaningful values (in the range :math:`[0,1]`).
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

    if not isinstance(dataset.GT, hl.CallExpression):
        raise Exception('GT field must be of type Call')

    if maf is not None:
        analyze('identity_by_descent/maf', maf, dataset._row_indices)
        dataset = dataset.select_rows(__maf=maf)
        dataset = dataset.filter_rows(hl.is_defined(dataset.__maf))
    else:
        dataset = dataset.select_rows()

    dataset = dataset.select_cols().select_globals().select_entries('GT')
    dataset = require_biallelic(dataset, 'ibd')

    if isinstance(Env.backend(), SparkBackend):
        return Table(
            ir.MatrixToTableApply(
                dataset._mir,
                {
                    'name': 'IBD',
                    'mafFieldName': '__maf' if maf is not None else None,
                    'bounded': bounded,
                    'min': min,
                    'max': max,
                },
            )
        ).persist()

    min = min or 0
    max = max or 1
    if not 0 <= min <= max <= 1:
        raise Exception(f"invalid pi hat filters {min} {max}")

    sample_ids = dataset.s.collect()
    if len(sample_ids) != len(set(sample_ids)):
        raise Exception('duplicate sample ids found')

    dataset = dataset.annotate_entries(
        n_alt_alleles=hl.or_else(dataset.GT.n_alt_alleles(), 0),
        is_hom_ref=hl.or_else(dataset.GT.is_hom_ref(), 0),
        is_het=hl.or_else(dataset.GT.is_het(), 0),
        is_hom_var=hl.or_else(dataset.GT.is_hom_var(), 0),
        is_missing=hl.is_missing(dataset.GT),
        is_not_missing=hl.is_defined(dataset.GT),
    )

    T = 2 * hl.agg.count_where(hl.is_defined(dataset.GT))
    X = hl.agg.sum(dataset.GT.n_alt_alleles())
    Y = T - X

    if maf is not None:
        p = dataset.__maf
    else:
        p = X / T

    q = 1 - p

    dataset = dataset.annotate_rows(
        _e00=(2 * (p**2) * (q**2) * ((X - 1) / X) * ((Y - 1) / Y) * (T / (T - 1)) * (T / (T - 2)) * (T / (T - 3))),
        _e10=(
            4 * (p**3) * q * ((X - 1) / X) * ((X - 2) / X) * (T / (T - 1)) * (T / (T - 2)) * (T / (T - 3))
            + 4 * p * (q**3) * ((Y - 1) / X) * ((Y - 2) / X) * (T / (T - 1)) * (T / (T - 2)) * (T / (T - 3))
        ),
        _e20=(
            (p**4) * ((X - 1) / X) * ((X - 2) / X) * ((X - 3) / X) * (T / (T - 1)) * (T / (T - 2)) * (T / (T - 3))
            + (q**4) * ((Y - 1) / Y) * ((Y - 2) / Y) * ((Y - 3) / Y) * (T / (T - 1)) * (T / (T - 2)) * (T / (T - 3))
            + 4 * (p**2) * (q**2) * ((X - 1) / X) * ((Y - 1) / Y) * (T / (T - 1)) * (T / (T - 2)) * (T / (T - 3))
        ),
        _e11=(
            2 * (p**2) * q * ((X - 1) / X) * (T / (T - 1)) * (T / (T - 2))
            + 2 * p * (q**2) * ((Y - 1) / Y) * (T / (T - 1)) * (T / (T - 2))
        ),
        _e21=(
            (p**3) * ((X - 1) / X) * ((X - 2) / X) * (T / (T - 1)) * (T / (T - 2))
            + (q**3) * ((Y - 1) / Y) * ((Y - 2) / Y) * (T / (T - 1)) * (T / (T - 2))
            + (p**2) * q * ((X - 1) / X) * (T / (T - 1)) * (T / (T - 2))
            + p * (q**2) * ((Y - 1) / Y) * (T / (T - 1)) * (T / (T - 2))
        ),
        _e22=(T / 2),
    )

    dataset = dataset.checkpoint(hl.utils.new_temp_file())

    expectations = dataset.aggregate_rows(
        hl.struct(
            e00=hl.agg.sum(dataset._e00),
            e10=hl.agg.sum(dataset._e10),
            e20=hl.agg.sum(dataset._e20),
            e11=hl.agg.sum(dataset._e11),
            e21=hl.agg.sum(dataset._e21),
            e22=hl.agg.sum(dataset._e22),
        )
    )

    IS_HOM_REF = BlockMatrix.from_entry_expr(dataset.is_hom_ref).checkpoint(hl.utils.new_temp_file())
    IS_HET = BlockMatrix.from_entry_expr(dataset.is_het).checkpoint(hl.utils.new_temp_file())
    IS_HOM_VAR = BlockMatrix.from_entry_expr(dataset.is_hom_var).checkpoint(hl.utils.new_temp_file())
    NOT_MISSING = (IS_HOM_REF + IS_HET + IS_HOM_VAR).checkpoint(hl.utils.new_temp_file())

    total_possible_ibs = NOT_MISSING.T @ NOT_MISSING

    ibs0_pre = (IS_HOM_REF.T @ IS_HOM_VAR).checkpoint(hl.utils.new_temp_file())
    ibs0 = ibs0_pre + ibs0_pre.T

    is_not_het = IS_HOM_REF + IS_HOM_VAR
    ibs1_pre = (IS_HET.T @ is_not_het).checkpoint(hl.utils.new_temp_file())
    ibs1 = ibs1_pre + ibs1_pre.T

    ibs2 = total_possible_ibs - ibs0 - ibs1

    Z0 = ibs0 / expectations.e00
    Z1 = (ibs1 - Z0 * expectations.e10) / expectations.e11
    Z2 = (ibs2 - Z0 * expectations.e20 - Z1 * expectations.e21) / expectations.e22

    def convert_to_table(bm, annotation_name):
        t = bm.entries()
        t = t.rename({'entry': annotation_name})
        return t

    z0 = convert_to_table(Z0, 'Z0').checkpoint(hl.utils.new_temp_file())
    z1 = convert_to_table(Z1, 'Z1').checkpoint(hl.utils.new_temp_file())
    z2 = convert_to_table(Z2, 'Z2').checkpoint(hl.utils.new_temp_file())
    ibs0 = convert_to_table(ibs0, 'ibs0').checkpoint(hl.utils.new_temp_file())
    ibs1 = convert_to_table(ibs1, 'ibs1').checkpoint(hl.utils.new_temp_file())
    ibs2 = convert_to_table(ibs2, 'ibs2').checkpoint(hl.utils.new_temp_file())

    result = z0.join(z1.join(z2).join(ibs0).join(ibs1).join(ibs2))

    def bound_result(_ibd):
        return (
            hl.case()
            .when(_ibd.Z0 > 1, hl.struct(Z0=hl.float(1), Z1=hl.float(0), Z2=hl.float(0)))
            .when(_ibd.Z1 > 1, hl.struct(Z0=hl.float(0), Z1=hl.float(1), Z2=hl.float(0)))
            .when(_ibd.Z2 > 1, hl.struct(Z0=hl.float(0), Z1=hl.float(0), Z2=hl.float(1)))
            .when(
                _ibd.Z0 < 0,
                hl.struct(Z0=hl.float(0), Z1=_ibd.Z1 / (_ibd.Z1 + _ibd.Z2), Z2=_ibd.Z2 / (_ibd.Z1 + _ibd.Z2)),
            )
            .when(
                _ibd.Z1 < 0,
                hl.struct(Z0=_ibd.Z0 / (_ibd.Z0 + _ibd.Z2), Z1=hl.float(0), Z2=_ibd.Z2 / (_ibd.Z0 + _ibd.Z2)),
            )
            .when(
                _ibd.Z2 < 0,
                hl.struct(Z0=_ibd.Z0 / (_ibd.Z0 + _ibd.Z1), Z1=_ibd.Z1 / (_ibd.Z0 + _ibd.Z1), Z2=hl.float(0)),
            )
            .default(_ibd)
        )

    result = result.annotate(ibd=hl.struct(Z0=result.Z0, Z1=result.Z1, Z2=result.Z2))
    result = result.drop('Z0', 'Z1', 'Z2')
    if bounded:
        result = result.annotate(ibd=bound_result(result.ibd))
    result = result.annotate(ibd=result.ibd.annotate(PI_HAT=result.ibd.Z1 / 2 + result.ibd.Z2))
    result = result.filter((result.i < result.j) & (min <= result.ibd.PI_HAT) & (result.ibd.PI_HAT <= max))

    samples = hl.literal(dataset.s.collect())
    result = result.key_by(i=samples[hl.int32(result.i)], j=samples[hl.int32(result.j)])

    return result.persist()
