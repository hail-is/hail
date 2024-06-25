from hail.expr.expressions import Float64Expression, expr_float64, expr_int32
from hail.expr.functions import _func
from hail.expr.types import tfloat64
from hail.typecheck import typecheck


@typecheck(ac=expr_int32, an=expr_int32, ci=expr_float64)
def filtering_allele_frequency(ac, an, ci) -> Float64Expression:
    """
    Computes a filtering allele frequency (described below)
    for `ac` and `an` with confidence `ci`.

    The filtering allele frequency is the highest true population allele frequency
    for which the upper bound of the `ci` (confidence interval) of allele count
    under a Poisson distribution is still less than the variant's observed
    `ac` (allele count) in the reference sample, given an `an` (allele number).

    This function defines a "filtering AF" that represents
    the threshold disease-specific "maximum credible AF" at or below which
    the disease could not plausibly be caused by that variant. A variant with
    a filtering AF >= the maximum credible AF for the disease under consideration
    should be filtered, while a variant with a filtering AF below the maximum
    credible remains a candidate. This filtering AF is not disease-specific:
    it can be applied to any disease of interest by comparing with a
    user-defined disease-specific maximum credible AF.

    For more details, see: `Whiffin et al., 2017 <https://www.nature.com/articles/gim201726>`__

    Parameters
    ----------
    ac : int or :class:`.Expression` of type :py:data:`.tint32`
    an : int or :class:`.Expression` of type :py:data:`.tint32`
    ci : float or :class:`.Expression` of type :py:data:`.tfloat64`

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return _func("filtering_allele_frequency", tfloat64, ac, an, ci)
