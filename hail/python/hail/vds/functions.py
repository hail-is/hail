import hail as hl
from hail.expr.expressions import expr_any, expr_array, expr_call, expr_int32
from hail.expr.functions import _func
from hail.typecheck import enumeration, typecheck


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
    """
    return hl.rbind(lgt, lambda lgt: hl.if_else(lgt.is_non_ref(), _func("lgt_to_gt", hl.tcall, lgt, la), lgt))


@typecheck(
    array=expr_array(),
    local_alleles=expr_array(expr_int32),
    n_alleles=expr_int32,
    fill_value=expr_any,
    number=enumeration('A', 'R', 'G'),
)
def local_to_global(array, local_alleles, n_alleles, fill_value, number):
    """Reindex a locally-indexed array to globally-indexed.

    Examples
    --------
    >>> local_alleles = hl.array([0, 2])
    >>> local_ad = hl.array([9, 10])
    >>> local_pl = hl.array([94, 0, 123])

    >>> hl.eval(local_to_global(local_ad, local_alleles, n_alleles=3, fill_value=0, number='R'))
    [9, 0, 10]

    >>> hl.eval(local_to_global(local_pl, local_alleles, n_alleles=3, fill_value=999, number='G'))
    [94, 999, 999, 0, 999, 123]

    Notes
    -----
    The `number` parameter matches the `VCF specification <https://samtools.github.io/hts-specs/VCFv4.3.pdf>`__
    number definitions:

     - ``A`` indicates one value per allele, excluding the reference.
     - ``R`` indicates one value per allele, including the reference.
     - ``G`` indicates one value per unique diploid genotype.

    Warning
    -------
    Using this function can lead to an enormous explosion in data size, without increasing
    information capacity. Its appropriate use is to conform to antiquated and badly-scaling
    representations (e.g. pVCF), but even so, caution should be exercised. Reindexing local
    PLs (or any G-numbered field) at a site with 1000 alleles will produce an array with
    more than 5,000 values per sample -- with 100,000 samples, nearly 50GB per variant!

    See Also
    --------
    :func:`.lgt_to_gt`

    Parameters
    ----------
    array : :class:`.ArrayExpression`
        Array to reindex.
    local_alleles : :class:`.ArrayExpression`
        Local alleles array.
    n_alleles : :class:`.Int32Expression`
        Total number of alleles to reindex to.
    fill_value
        Value to fill in at global indices with no data.
    number : :class:`str`
        One of 'A', 'R', 'G'.

    Returns
    -------
    :class:`.ArrayExpression`
    """
    try:
        fill_value = hl.coercer_from_dtype(array.dtype.element_type).coerce(fill_value)
    except Exception as e:
        raise ValueError(f'fill_value type {fill_value.dtype} is incompatible with array type {array.dtype}') from e

    if number == 'G':
        return _func("local_to_global_g", array.dtype, array, local_alleles, n_alleles, fill_value)
    elif number == 'R':
        omit_first = False
    elif number == 'A':
        omit_first = True
    else:
        raise ValueError(f'unrecognized number {number}')

    return _func("local_to_global_a_r", array.dtype, array, local_alleles, n_alleles, fill_value, hl.bool(omit_first))
