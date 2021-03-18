import hail as hl

from hail.expr.expressions import expr_call
from hail.expr.expressions import matrix_table_source
from hail.typecheck import typecheck, nullable
from hail.utils import deduplicate
from hail.utils.java import Env


@typecheck(call_expr=expr_call, block_size=nullable(int))
def king(call_expr, *, block_size=None):
    r"""Compute relatedness estimates between individuals using a KING variant.

    .. include:: ../_templates/req_diploid_gt.rst

    Examples
    --------
    Estimate the kinship coefficient for every pair of samples.

    >>> kinship = hl.king(dataset.GT)

    Notes
    -----

    The following presentation summarizes the methods section of `Manichaikul,
    et. al. <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3025716/>`__, but
    adopts a more consistent notation for matrices.

    Let

    - :math:`i` and :math:`j` be two individuals in the dataset

    - :math:`N^{Aa}_{i}` be the number of heterozygote genotypes for individual
      :math:`i`.

    - :math:`N^{Aa,Aa}_{i,j}` be the number of variants at which a pair of
      individuals both have heterozygote genotypes.

    - :math:`N^{AA,aa}_{i,j}` be the number of variants at which a pair of
      individuals have opposing homozygote genotypes.

    - :math:`S_{i,j}` be the set of single-nucleotide variants for which both
      individuals :math:`i` and :math:`j` have a non-missing genotype.

    - :math:`X_{i,s}` be the genotype score matrix. Each entry corresponds to
      the genotype of individual :math:`i` at variant
      :math:`s`. Homozygous-reference genotypes are represented as 0,
      heterozygous genotypes are represented as 1, and homozygous-alternate
      genotypes are represented as 2. :math:`X_{i,s}` is calculated by invoking
      :meth:`~.CallExpression.n_alt_alleles` on the `call_expr`.

    The three counts above, :math:`N^{Aa}`, :math:`N^{Aa,Aa}`, and
    :math:`N^{AA,aa}`, exclude variants where one or both individuals have
    missing genotypes.

    In terms of the symbols above, we can define :math:`d`, the genetic distance
    between two samples. We can interpret :math:`d` as an unnormalized
    measurement of the genetic material not shared identically-by-descent:

    .. math::

       d_{i,j} = \sum_{s \in S_{i,j}}\left(X_{i,s} - X_{j,s}\right)^2

    In the supplement to Manichaikul, et. al, the authors show how to re-express
    the genetic distance above in terms of the three counts of hetero- and
    homozygosity by considering the nine possible configurations of a pair of
    genotypes:

    +-------------------------------+----------+----------+----------+
    |:math:`(X_{i,s} - X_{j,s})^2`  |homref    |het       |homalt    |
    +-------------------------------+----------+----------+----------+
    |homref                         |0         |1         |4         |
    +-------------------------------+----------+----------+----------+
    |het                            |1         |0         |1         |
    +-------------------------------+----------+----------+----------+
    |homalt                         |4         |1         |0         |
    +-------------------------------+----------+----------+----------+

    which leads to this expression for genetic distance:

    .. math::

        d_{i,j} = 4 N^{AA,aa}_{i,j}
                  + N^{Aa}_{i}
                  + N^{Aa}_{j}
                  - 2 N^{Aa,Aa}_{i,j}

    The first term, :math:`4 N^{AA,aa}_{i,j}`, accounts for all pairs of
    genotypes with opposing homozygous genotypes. The second and third terms
    account for the four cases of one heteroyzgous genotype and one
    non-heterozygous genotype. Unfortunately, the second and third term also
    contribute to the case of a pair of heteroyzgous genotypes. We offset this
    with the fourth and final term.

    The genetic distance, :math:`d_{i,j}`, ranges between zero and four times
    the number of variants in the dataset. In the supplement to Manichaikul,
    et. al, the authors demonstrate that the kinship coefficient,
    :math:`\phi_{i,j}`, between two individuals from the same population is
    related to the expected genetic distance at any *one* variant by way of the
    allele frequency:

    .. math::

        \mathop{\mathbb{E}}_{i,j} (X_{i,s} - X_{j,s})^2 =
            4 p_s (1 - p_s) (1 - 2\phi_{i,j})

    This identity reveals that the quotient of the expected genetic distance and
    the four-trial binomial variance in the allele frequency represents,
    roughly, the "fraction of genetic material *not* shared
    identically-by-descent":

    .. math::

        1 - 2 \phi_{i,j} = \frac{4 N^{AA,aa}_{i,j}
                                 + N^{Aa}_{i}
                                 + N^{Aa}_{j}
                                 - 2 N^{Aa,Aa}_{i,j}}
                                {\sum_{s \in S_{i,j}} 4 p_s (1 - p_s)}

    Note that the "coefficient of relationship", (by definition: the fraction of
    genetic material shared identically-by-descent) is equal to twice the
    kinship coefficient: :math:`\phi_{i,j} = 2 r_{i,j}`.

    Manichaikul, et. al, assuming one homogeneous population, demonstrate in
    Section 2.3 that the sum of the variance of the allele frequencies,

    .. math::

        \sum_{s \in S_{i, j}} 2 p_s (1 - p_s)

    is, in expectation, proportional to the count of heterozygous genotypes of
    either individual:

    .. math::

        N^{Aa}_{i}

    For individuals from distinct populations, the authors propose replacing the
    count of heteroyzgous genotypes with the average of the two individuals:

    .. math::

        \frac{N^{Aa}_{i} + N^{Aa}_{j}}{2}

    Using the aforementioned equality, we define a normalized genetic distance,
    :math:`\widetilde{d_{i,j}}`, for a pair of individuals from distinct
    populations:

    .. math::

        \begin{aligned}
        \widetilde{d_{i,j}} &=
            \frac{4 N^{AA,aa}_{i,j} + N^{Aa}_{i} + N^{Aa}_{j} - 2 N^{Aa,Aa}_{i,j}}
                 {N^{Aa}_{i} + N^{Aa}_{j}} \\
            &= 1
               + \frac{4 N^{AA,aa}_{i,j} - 2 N^{Aa,Aa}_{i,j}}
                      {N^{Aa}_{i} + N^{Aa}_{j}}
        \end{aligned}

    As mentioned before, the complement of the normalized genetic distance is
    the coefficient of relationship which is also equal to twice the kinship
    coefficient:

    .. math::

        2 \phi_{i,j} = r_{i,j} = 1 - \widetilde{d_{i,j}}

    We now present the KING "within-family" estimator of the kinship coefficient
    as one-half of the coefficient of relationship:

    .. math::

        \begin{aligned}
        \widehat{\phi_{i,j}^{\mathrm{within}}} &= \frac{1}{2} r_{i,j} \\
            &= \frac{1}{2} \left( 1 - \widetilde{d_{i,j}} \right) \\
            &= \frac{N^{Aa,Aa}_{i,j} - 2 N^{AA,aa}_{i,j}}
                    {N^{Aa}_{i} + N^{Aa}_{j}}
        \end{aligned}

    This "within-family" estimator over-estimates the kinship coefficient under
    certain circumstances detailed in Section 2.3 of Manichaikul, et. al. The
    authors recommend an alternative estimator when individuals are known to be
    from different families. The estimator replaces the average count of
    heteroyzgous genotypes with the minimum count of heterozygous genotypes:

    .. math::

        \frac{N^{Aa}_{i} + N^{Aa}_{j}}{2} \rightsquigarrow \mathrm{min}(N^{Aa}_{i}, N^{Aa}_{j})

    This transforms the "within-family" estimator into the "between-family"
    estimator, defined by Equation 11 of Manichaikul, et. al.:

    .. math::

        \begin{aligned}
        \widetilde{d_{i,j}^{\mathrm{between}}} &=
            \frac{4 N^{AA,aa}_{i,j} + N^{Aa}_{i} + N^{Aa}_{j} - 2 N^{Aa,Aa}_{i,j}}
                 {2 \mathrm{min}(N^{Aa}_{i}, N^{Aa}_{j})} \\
        \widehat{\phi_{i,j}^{\mathrm{between}}} &=
            \frac{1}{2}
            + \frac{2 N^{Aa,Aa}_{i,j} - 4 N^{AA,aa}_{i,j} - N^{Aa}_{i} - N^{Aa}_{j}}
                   {4 \cdot \mathrm{min}(N^{Aa}_{i}, N^{Aa}_{j})}
        \end{aligned}

    This function, :func:`.king`, only implements the "between-family"
    estimator, :math:`\widehat{\phi_{i,j}^{\mathrm{between}}}`.

    Parameters
    ----------
    call_expr : :class:`.CallExpression`
        Entry-indexed call expression.
    block_size : :obj:`int`, optional
        Block size of block matrices used in the algorithm.
        Default given by :meth:`.BlockMatrix.default_block_size`.

    Returns
    -------
    :class:`.MatrixTable`
        A :class:`.MatrixTable` whose rows and columns are keys are taken from
       `call-expr`'s column keys. It has one entry field, `phi`.
    """
    mt = matrix_table_source('king/call_expr', call_expr)
    call = Env.get_uid()
    mt = mt.annotate_entries(**{call: call_expr})

    is_hom_ref = Env.get_uid()
    is_het = Env.get_uid()
    is_hom_var = Env.get_uid()
    is_defined = Env.get_uid()
    mt = mt.unfilter_entries()
    mt = mt.select_entries(**{
        is_hom_ref: hl.float(hl.or_else(mt[call].is_hom_ref(), 0)),
        is_het: hl.float(hl.or_else(mt[call].is_het(), 0)),
        is_hom_var: hl.float(hl.or_else(mt[call].is_hom_var(), 0)),
        is_defined: hl.float(hl.is_defined(mt[call]))
    })
    ref = hl.linalg.BlockMatrix.from_entry_expr(mt[is_hom_ref], block_size=block_size)
    het = hl.linalg.BlockMatrix.from_entry_expr(mt[is_het], block_size=block_size)
    var = hl.linalg.BlockMatrix.from_entry_expr(mt[is_hom_var], block_size=block_size)
    defined = hl.linalg.BlockMatrix.from_entry_expr(mt[is_defined], block_size=block_size)
    ref_var = (ref.T @ var).checkpoint(hl.utils.new_temp_file())
    # We need the count of times the pair is AA,aa and aa,AA. ref_var is only
    # AA,aa.  Transposing ref_var gives var_ref, i.e. aa,AA.
    #
    # n.b. (REF.T @ VAR).T == (VAR.T @ REF) by laws of matrix multiply
    N_AA_aa = ref_var + ref_var.T
    N_Aa_Aa = (het.T @ het).checkpoint(hl.utils.new_temp_file())
    # We count the times the row individual has a heterozygous genotype and the
    # column individual has any defined genotype at all.
    N_Aa_defined = (het.T @ defined).checkpoint(hl.utils.new_temp_file())

    het_hom_balance = N_Aa_Aa - (2 * N_AA_aa)
    het_hom_balance = het_hom_balance.to_matrix_table_row_major()
    n_hets_for_rows = N_Aa_defined.to_matrix_table_row_major()
    n_hets_for_cols = N_Aa_defined.T.to_matrix_table_row_major()

    kinship_between = het_hom_balance.rename({'element': 'het_hom_balance'})
    kinship_between = kinship_between.annotate_entries(
        n_hets_row=n_hets_for_rows[kinship_between.row_key, kinship_between.col_key].element,
        n_hets_col=n_hets_for_cols[kinship_between.row_key, kinship_between.col_key].element
    )

    col_index_field = Env.get_uid()
    col_key = mt.col_key
    cols = mt.add_col_index(col_index_field).key_cols_by(col_index_field).cols()

    kinship_between = kinship_between.key_cols_by(
        **cols[kinship_between.col_idx].select(*col_key)
    )

    renaming, _ = deduplicate(list(col_key), already_used=set(col_key))
    assert len(renaming) == len(col_key)

    kinship_between = kinship_between.key_rows_by(
        **cols[kinship_between.row_idx].select(*col_key).rename(dict(renaming))
    )

    kinship_between = kinship_between.annotate_entries(
        min_n_hets=hl.min(kinship_between.n_hets_row,
                          kinship_between.n_hets_col)
    )
    return kinship_between.select_entries(
        phi=(
            0.5
        ) + (
            (
                2 * kinship_between.het_hom_balance +
                - kinship_between.n_hets_row
                - kinship_between.n_hets_col
            ) / (
                4 * kinship_between.min_n_hets
            )
        )
    ).select_rows().select_cols().select_globals()
