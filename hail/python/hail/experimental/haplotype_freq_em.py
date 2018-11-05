from hail.expr.expressions import *
from hail.expr.expressions.expression_typecheck import *
from hail.expr.types import *
from hail.typecheck import *
from hail.expr.functions import _func


@typecheck(gt_counts=expr_array(expr_int32))
def haplotype_freq_em(gt_counts) -> ArrayExpression:
    """
    Computes estimated haplotype counts based on genotypes for a pair of bi-allelic variants.
    Implements the Excoffier & Slatkin EM (Exccoffier & Slatkin, Mol. Biol. Evol. 1995)

    The unphased input genotype counts for the variant pairs has to be provided in the following order:
    [AABB, AABb, AAbb, AaBB, AaBb, Aabb, aaBB, aaBb, aabb]

    The estimated haplotype counts are returned in an array in the following order:
    [AB, Ab, aB, ab]

    Where _A_ and _a_ are the reference and non-reference alleles for the first variant, resp.
    And _B_ and _b_ are the reference and non-reference alleles for the second variant, resp.

    Parameters
    ----------
    gt_counts : :class:`.ArrayExpression`

    Returns
    -------
    :class:`.ArrayExpression`
    """
    return _func("haplotype_freq_em", tarray(tfloat64), gt_counts)