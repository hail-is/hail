import hail as hl
from hail.expr.expressions import expr_array, expr_call, expr_int32
from hail.typecheck import typecheck


@typecheck(lgt=expr_call, la=expr_array(expr_int32))
def lgt_to_gt(lgt, la):
    """Transform LGT into GT using local alleles array.

    Parameters
    ----------
    lgt : :class:`.CallExpression`
        LGT value.
    la : :class:`.ArrayExpression`
        Local alleles array.

    Returns
    -------
    :class:`.CallExpression`

    Notes
    -----
    This function assumes diploid genotypes.
    """
    return hl.call(la[lgt[0]], la[lgt[1]], phased=lgt.phased)
