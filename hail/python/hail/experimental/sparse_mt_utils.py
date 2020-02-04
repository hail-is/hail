from hail import call
from hail.expr.expressions import expr_array, expr_call, expr_int32
from hail.typecheck import typecheck


@typecheck(lgt=expr_call, la=expr_array(expr_int32))
def lgt_to_gt(lgt, la):
    """Transforming Local GT and Local Alleles into the true GT

    Parameters
    ----------
    lgt : :class:`.CallExpression`
        The LGT value
    la : :class:`.ArrayExpression`
        The Local Alleles array

    Returns
    -------
    :class:`.CallExpression`

    Notes
    -----
    This function assumes diploid genotypes.
    """
    return call(la[lgt[0]], la[lgt[1]])
