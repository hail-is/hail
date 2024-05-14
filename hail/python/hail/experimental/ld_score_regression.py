import hail as hl
from hail.expr.expressions import expr_float64, expr_numeric, analyze
from hail.typecheck import typecheck, oneof, sequenceof, nullable
from hail.utils import wrap_to_list, new_temp_file


@typecheck(
    weight_expr=expr_float64,
    ld_score_expr=expr_numeric,
    chi_sq_exprs=oneof(expr_float64, sequenceof(expr_float64)),
    n_samples_exprs=oneof(expr_numeric, sequenceof(expr_numeric)),
    n_blocks=int,
    two_step_threshold=int,
    n_reference_panel_variants=nullable(int),
)
def ld_score_regression(
    weight_expr,
    ld_score_expr,
    chi_sq_exprs,
    n_samples_exprs,
    n_blocks=200,
    two_step_threshold=30,
    n_reference_panel_variants=None,
) -> hl.Table:
    r"""Estimate SNP-heritability and level of confounding biases from genome-wide association study
    (GWAS) summary statistics.

    Given a set or multiple sets of GWAS summary statistics, :func:`.ld_score_regression` estimates the heritability
    of a trait or set of traits and the level of confounding biases present in
    the underlying studies by regressing chi-squared statistics on LD scores,
    leveraging the model:

    .. math::

        \mathrm{E}[\chi_j^2] = 1 + Na + \frac{Nh_g^2}{M}l_j

    *  :math:`\mathrm{E}[\chi_j^2]` is the expected chi-squared statistic
       for variant :math:`j` resulting from a test of association between
       variant :math:`j` and a trait.
    *  :math:`l_j = \sum_{k} r_{jk}^2` is the LD score of variant
       :math:`j`, calculated as the sum of squared correlation coefficients
       between variant :math:`j` and nearby variants. See :func:`ld_score`
       for further details.
    *  :math:`a` captures the contribution of confounding biases, such as
       cryptic relatedness and uncontrolled population structure, to the
       association test statistic.
    *  :math:`h_g^2` is the SNP-heritability, or the proportion of variation
       in the trait explained by the effects of variants included in the
       regression model above.
    *  :math:`M` is the number of variants used to estimate :math:`h_g^2`.
    *  :math:`N` is the number of samples in the underlying association study.

    For more details on the method implemented in this function, see:

    * `LD Score regression distinguishes confounding from polygenicity in genome-wide association studies (Bulik-Sullivan et al, 2015) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4495769/>`__

    Examples
    --------

    Run the method on a matrix table of summary statistics, where the rows
    are variants and the columns are different phenotypes:

    >>> mt_gwas = ld_score_all_phenos_sumstats
    >>> ht_results = hl.experimental.ld_score_regression(
    ...     weight_expr=mt_gwas['ld_score'],
    ...     ld_score_expr=mt_gwas['ld_score'],
    ...     chi_sq_exprs=mt_gwas['chi_squared'],
    ...     n_samples_exprs=mt_gwas['n'])


    Run the method on a table with summary statistics for a single
    phenotype:

    >>> ht_gwas = ld_score_one_pheno_sumstats
    >>> ht_results = hl.experimental.ld_score_regression(
    ...     weight_expr=ht_gwas['ld_score'],
    ...     ld_score_expr=ht_gwas['ld_score'],
    ...     chi_sq_exprs=ht_gwas['chi_squared_50_irnt'],
    ...     n_samples_exprs=ht_gwas['n_50_irnt'])

    Run the method on a table with summary statistics for multiple
    phenotypes:

    >>> ht_gwas = ld_score_one_pheno_sumstats
    >>> ht_results = hl.experimental.ld_score_regression(
    ...     weight_expr=ht_gwas['ld_score'],
    ...     ld_score_expr=ht_gwas['ld_score'],
    ...     chi_sq_exprs=[ht_gwas['chi_squared_50_irnt'],
    ...                        ht_gwas['chi_squared_20160']],
    ...     n_samples_exprs=[ht_gwas['n_50_irnt'],
    ...                      ht_gwas['n_20160']])

    Notes
    -----
    The ``exprs`` provided as arguments to :func:`.ld_score_regression`
    must all be from the same object, either a :class:`~.Table` or a
    :class:`~.MatrixTable`.

    **If the arguments originate from a table:**

    *  The table must be keyed by fields ``locus`` of type
       :class:`.tlocus` and ``alleles``, a :class:`.tarray` of
       :py:data:`.tstr` elements.
    *  ``weight_expr``, ``ld_score_expr``, ``chi_sq_exprs``, and
       ``n_samples_exprs`` are must be row-indexed fields.
    *  The number of expressions passed to ``n_samples_exprs`` must be
       equal to one or the number of expressions passed to
       ``chi_sq_exprs``. If just one expression is passed to
       ``n_samples_exprs``, that sample size expression is assumed to
       apply to all sets of statistics passed to ``chi_sq_exprs``.
       Otherwise, the expressions passed to ``chi_sq_exprs`` and
       ``n_samples_exprs`` are matched by index.
    *  The ``phenotype`` field that keys the table returned by
       :func:`.ld_score_regression` will have generic :obj:`int` values
       ``0``, ``1``, etc. corresponding to the ``0th``, ``1st``, etc.
       expressions passed to the ``chi_sq_exprs`` argument.

    **If the arguments originate from a matrix table:**

    *  The dimensions of the matrix table must be variants
       (rows) by phenotypes (columns).
    *  The rows of the matrix table must be keyed by fields
       ``locus`` of type :class:`.tlocus` and ``alleles``,
       a :class:`.tarray` of :py:data:`.tstr` elements.
    *  The columns of the matrix table must be keyed by a field
       of type :py:data:`.tstr` that uniquely identifies phenotypes
       represented in the matrix table. The column key must be a single
       expression; compound keys are not accepted.
    *  ``weight_expr`` and ``ld_score_expr`` must be row-indexed
       fields.
    *  ``chi_sq_exprs`` must be a single entry-indexed field
       (not a list of fields).
    *  ``n_samples_exprs`` must be a single entry-indexed field
       (not a list of fields).
    *  The ``phenotype`` field that keys the table returned by
       :func:`.ld_score_regression` will have values corresponding to the
       column keys of the input matrix table.

    This function returns a :class:`~.Table` with one row per set of summary
    statistics passed to the ``chi_sq_exprs`` argument. The following
    row-indexed fields are included in the table:

    *  **phenotype** (:py:data:`.tstr`) -- The name of the phenotype. The
       returned table is keyed by this field. See the notes below for
       details on the possible values of this field.
    *  **mean_chi_sq** (:py:data:`.tfloat64`) -- The mean chi-squared
       test statistic for the given phenotype.
    *  **intercept** (`Struct`) -- Contains fields:

       -  **estimate** (:py:data:`.tfloat64`) -- A point estimate of the
          intercept :math:`1 + Na`.
       -  **standard_error**  (:py:data:`.tfloat64`) -- An estimate of
          the standard error of this point estimate.

    *  **snp_heritability** (`Struct`) -- Contains fields:

       -  **estimate** (:py:data:`.tfloat64`) -- A point estimate of the
          SNP-heritability :math:`h_g^2`.
       -  **standard_error** (:py:data:`.tfloat64`) -- An estimate of
          the standard error of this point estimate.

    Warning
    -------
    :func:`.ld_score_regression` considers only the rows for which both row
    fields ``weight_expr`` and ``ld_score_expr`` are defined. Rows with missing
    values in either field are removed prior to fitting the LD score
    regression model.

    Parameters
    ----------
    weight_expr : :class:`.Float64Expression`
                  Row-indexed expression for the LD scores used to derive
                  variant weights in the model.
    ld_score_expr : :class:`.Float64Expression`
                    Row-indexed expression for the LD scores used as covariates
                    in the model.
    chi_sq_exprs : :class:`.Float64Expression` or :obj:`list` of
                        :class:`.Float64Expression`
                        One or more row-indexed (if table) or entry-indexed
                        (if matrix table) expressions for chi-squared
                        statistics resulting from genome-wide association
                        studies (GWAS).
    n_samples_exprs: :class:`.NumericExpression` or :obj:`list` of
                     :class:`.NumericExpression`
                     One or more row-indexed (if table) or entry-indexed
                     (if matrix table) expressions indicating the number of
                     samples used in the studies that generated the test
                     statistics supplied to ``chi_sq_exprs``.
    n_blocks : :obj:`int`
               The number of blocks used in the jackknife approach to
               estimating standard errors.
    two_step_threshold : :obj:`int`
                         Variants with chi-squared statistics greater than this
                         value are excluded in the first step of the two-step
                         procedure used to fit the model.
    n_reference_panel_variants : :obj:`int`, optional
                                 Number of variants used to estimate the
                                 SNP-heritability :math:`h_g^2`.

    Returns
    -------
    :class:`~.Table`
        Table keyed by ``phenotype`` with intercept and heritability estimates
        for each phenotype passed to the function."""

    chi_sq_exprs = wrap_to_list(chi_sq_exprs)
    n_samples_exprs = wrap_to_list(n_samples_exprs)

    assert (len(chi_sq_exprs) == len(n_samples_exprs)) or (len(n_samples_exprs) == 1)
    __k = 2  # number of covariates, including intercept

    ds = chi_sq_exprs[0]._indices.source

    analyze('ld_score_regression/weight_expr', weight_expr, ds._row_indices)
    analyze('ld_score_regression/ld_score_expr', ld_score_expr, ds._row_indices)

    # format input dataset
    if isinstance(ds, hl.MatrixTable):
        if len(chi_sq_exprs) != 1:
            raise ValueError("""Only one chi_sq_expr allowed if originating
                from a matrix table.""")
        if len(n_samples_exprs) != 1:
            raise ValueError("""Only one n_samples_expr allowed if
                originating from a matrix table.""")

        col_key = list(ds.col_key)
        if len(col_key) != 1:
            raise ValueError("""Matrix table must be keyed by a single
                phenotype field.""")

        analyze('ld_score_regression/chi_squared_expr', chi_sq_exprs[0], ds._entry_indices)
        analyze('ld_score_regression/n_samples_expr', n_samples_exprs[0], ds._entry_indices)

        ds = ds._select_all(
            row_exprs={
                '__locus': ds.locus,
                '__alleles': ds.alleles,
                '__w_initial': weight_expr,
                '__w_initial_floor': hl.max(weight_expr, 1.0),
                '__x': ld_score_expr,
                '__x_floor': hl.max(ld_score_expr, 1.0),
            },
            row_key=['__locus', '__alleles'],
            col_exprs={'__y_name': ds[col_key[0]]},
            col_key=['__y_name'],
            entry_exprs={'__y': chi_sq_exprs[0], '__n': n_samples_exprs[0]},
        )
        ds = ds.annotate_entries(**{'__w': ds.__w_initial})

        ds = ds.filter_rows(
            hl.is_defined(ds.__locus)
            & hl.is_defined(ds.__alleles)
            & hl.is_defined(ds.__w_initial)
            & hl.is_defined(ds.__x)
        )

    else:
        assert isinstance(ds, hl.Table)
        for y in chi_sq_exprs:
            analyze('ld_score_regression/chi_squared_expr', y, ds._row_indices)
        for n in n_samples_exprs:
            analyze('ld_score_regression/n_samples_expr', n, ds._row_indices)

        ys = ['__y{:}'.format(i) for i, _ in enumerate(chi_sq_exprs)]
        ws = ['__w{:}'.format(i) for i, _ in enumerate(chi_sq_exprs)]
        ns = ['__n{:}'.format(i) for i, _ in enumerate(n_samples_exprs)]

        ds = ds.select(
            **dict(
                **{'__locus': ds.locus, '__alleles': ds.alleles, '__w_initial': weight_expr, '__x': ld_score_expr},
                **{y: chi_sq_exprs[i] for i, y in enumerate(ys)},
                **{w: weight_expr for w in ws},
                **{n: n_samples_exprs[i] for i, n in enumerate(ns)},
            )
        )
        ds = ds.key_by(ds.__locus, ds.__alleles)

        table_tmp_file = new_temp_file()
        ds.write(table_tmp_file)
        ds = hl.read_table(table_tmp_file)

        hts = [
            ds.select(**{
                '__w_initial': ds.__w_initial,
                '__w_initial_floor': hl.max(ds.__w_initial, 1.0),
                '__x': ds.__x,
                '__x_floor': hl.max(ds.__x, 1.0),
                '__y_name': i,
                '__y': ds[ys[i]],
                '__w': ds[ws[i]],
                '__n': hl.int(ds[ns[i]]),
            })
            for i, y in enumerate(ys)
        ]

        mts = [
            ht.to_matrix_table(
                row_key=['__locus', '__alleles'],
                col_key=['__y_name'],
                row_fields=['__w_initial', '__w_initial_floor', '__x', '__x_floor'],
            )
            for ht in hts
        ]

        ds = mts[0]
        for i in range(1, len(ys)):
            ds = ds.union_cols(mts[i])

        ds = ds.filter_rows(
            hl.is_defined(ds.__locus)
            & hl.is_defined(ds.__alleles)
            & hl.is_defined(ds.__w_initial)
            & hl.is_defined(ds.__x)
        )

    mt_tmp_file1 = new_temp_file()
    ds.write(mt_tmp_file1)
    mt = hl.read_matrix_table(mt_tmp_file1)

    if not n_reference_panel_variants:
        M = mt.count_rows()
    else:
        M = n_reference_panel_variants

    mt = mt.annotate_entries(
        __in_step1=(hl.is_defined(mt.__y) & (mt.__y < two_step_threshold)), __in_step2=hl.is_defined(mt.__y)
    )

    mt = mt.annotate_cols(
        __col_idx=hl.int(hl.scan.count()),
        __m_step1=hl.agg.count_where(mt.__in_step1),
        __m_step2=hl.agg.count_where(mt.__in_step2),
    )

    col_keys = list(mt.col_key)

    ht = mt.localize_entries(entries_array_field_name='__entries', columns_array_field_name='__cols')

    ht = ht.annotate(
        __entries=hl.rbind(
            hl.scan.array_agg(lambda entry: hl.scan.count_where(entry.__in_step1), ht.__entries),
            lambda step1_indices: hl.map(
                lambda i: hl.rbind(
                    hl.int(hl.or_else(step1_indices[i], 0)),
                    ht.__cols[i].__m_step1,
                    ht.__entries[i],
                    lambda step1_idx, m_step1, entry: hl.rbind(
                        hl.map(lambda j: hl.int(hl.floor(j * (m_step1 / n_blocks))), hl.range(0, n_blocks + 1)),
                        lambda step1_separators: hl.rbind(
                            hl.set(step1_separators).contains(step1_idx),
                            hl.sum(hl.map(lambda s1: step1_idx >= s1, step1_separators)) - 1,
                            lambda is_separator, step1_block: entry.annotate(
                                __step1_block=step1_block,
                                __step2_block=hl.if_else(
                                    ~entry.__in_step1 & is_separator, step1_block - 1, step1_block
                                ),
                            ),
                        ),
                    ),
                ),
                hl.range(0, hl.len(ht.__entries)),
            ),
        )
    )

    mt = ht._unlocalize_entries('__entries', '__cols', col_keys)

    mt_tmp_file2 = new_temp_file()
    mt.write(mt_tmp_file2)
    mt = hl.read_matrix_table(mt_tmp_file2)

    # initial coefficient estimates
    mt = mt.annotate_cols(__initial_betas=[1.0, (hl.agg.mean(mt.__y) - 1.0) / hl.agg.mean(mt.__x)])
    mt = mt.annotate_cols(__step1_betas=mt.__initial_betas, __step2_betas=mt.__initial_betas)

    # step 1 iteratively reweighted least squares
    for i in range(3):
        mt = mt.annotate_entries(
            __w=hl.if_else(
                mt.__in_step1,
                1.0 / (mt.__w_initial_floor * 2.0 * (mt.__step1_betas[0] + mt.__step1_betas[1] * mt.__x_floor) ** 2),
                0.0,
            )
        )
        mt = mt.annotate_cols(
            __step1_betas=hl.agg.filter(mt.__in_step1, hl.agg.linreg(y=mt.__y, x=[1.0, mt.__x], weight=mt.__w).beta)
        )
        mt = mt.annotate_cols(__step1_h2=hl.max(hl.min(mt.__step1_betas[1] * M / hl.agg.mean(mt.__n), 1.0), 0.0))
        mt = mt.annotate_cols(__step1_betas=[mt.__step1_betas[0], mt.__step1_h2 * hl.agg.mean(mt.__n) / M])

    # step 1 block jackknife
    mt = mt.annotate_cols(
        __step1_block_betas=hl.agg.array_agg(
            lambda i: hl.agg.filter(
                (mt.__step1_block != i) & mt.__in_step1, hl.agg.linreg(y=mt.__y, x=[1.0, mt.__x], weight=mt.__w).beta
            ),
            hl.range(n_blocks),
        )
    )

    mt = mt.annotate_cols(
        __step1_block_betas_bias_corrected=hl.map(
            lambda x: n_blocks * mt.__step1_betas - (n_blocks - 1) * x, mt.__step1_block_betas
        )
    )

    mt = mt.annotate_cols(
        __step1_jackknife_mean=hl.map(
            lambda i: hl.mean(hl.map(lambda x: x[i], mt.__step1_block_betas_bias_corrected)), hl.range(0, __k)
        ),
        __step1_jackknife_variance=hl.map(
            lambda i: (
                hl.sum(hl.map(lambda x: x[i] ** 2, mt.__step1_block_betas_bias_corrected))
                - hl.sum(hl.map(lambda x: x[i], mt.__step1_block_betas_bias_corrected)) ** 2 / n_blocks
            )
            / (n_blocks - 1)
            / n_blocks,
            hl.range(0, __k),
        ),
    )

    # step 2 iteratively reweighted least squares
    for i in range(3):
        mt = mt.annotate_entries(
            __w=hl.if_else(
                mt.__in_step2,
                1.0 / (mt.__w_initial_floor * 2.0 * (mt.__step2_betas[0] + +mt.__step2_betas[1] * mt.__x_floor) ** 2),
                0.0,
            )
        )
        mt = mt.annotate_cols(
            __step2_betas=[
                mt.__step1_betas[0],
                hl.agg.filter(
                    mt.__in_step2, hl.agg.linreg(y=mt.__y - mt.__step1_betas[0], x=[mt.__x], weight=mt.__w).beta[0]
                ),
            ]
        )
        mt = mt.annotate_cols(__step2_h2=hl.max(hl.min(mt.__step2_betas[1] * M / hl.agg.mean(mt.__n), 1.0), 0.0))
        mt = mt.annotate_cols(__step2_betas=[mt.__step1_betas[0], mt.__step2_h2 * hl.agg.mean(mt.__n) / M])

    # step 2 block jackknife
    mt = mt.annotate_cols(
        __step2_block_betas=hl.agg.array_agg(
            lambda i: hl.agg.filter(
                (mt.__step2_block != i) & mt.__in_step2,
                hl.agg.linreg(y=mt.__y - mt.__step1_betas[0], x=[mt.__x], weight=mt.__w).beta[0],
            ),
            hl.range(n_blocks),
        )
    )

    mt = mt.annotate_cols(
        __step2_block_betas_bias_corrected=hl.map(
            lambda x: n_blocks * mt.__step2_betas[1] - (n_blocks - 1) * x, mt.__step2_block_betas
        )
    )

    mt = mt.annotate_cols(
        __step2_jackknife_mean=hl.mean(mt.__step2_block_betas_bias_corrected),
        __step2_jackknife_variance=(
            hl.sum(mt.__step2_block_betas_bias_corrected**2)
            - hl.sum(mt.__step2_block_betas_bias_corrected) ** 2 / n_blocks
        )
        / (n_blocks - 1)
        / n_blocks,
    )

    # combine step 1 and step 2 block jackknifes
    mt = mt.annotate_entries(
        __step2_initial_w=1.0
        / (mt.__w_initial_floor * 2.0 * (mt.__initial_betas[0] + +mt.__initial_betas[1] * mt.__x_floor) ** 2)
    )

    mt = mt.annotate_cols(
        __final_betas=[mt.__step1_betas[0], mt.__step2_betas[1]],
        __c=(hl.agg.sum(mt.__step2_initial_w * mt.__x) / hl.agg.sum(mt.__step2_initial_w * mt.__x**2)),
    )

    mt = mt.annotate_cols(
        __final_block_betas=hl.map(
            lambda i: (mt.__step2_block_betas[i] - mt.__c * (mt.__step1_block_betas[i][0] - mt.__final_betas[0])),
            hl.range(0, n_blocks),
        )
    )

    mt = mt.annotate_cols(
        __final_block_betas_bias_corrected=(n_blocks * mt.__final_betas[1] - (n_blocks - 1) * mt.__final_block_betas)
    )

    mt = mt.annotate_cols(
        __final_jackknife_mean=[mt.__step1_jackknife_mean[0], hl.mean(mt.__final_block_betas_bias_corrected)],
        __final_jackknife_variance=[
            mt.__step1_jackknife_variance[0],
            (
                hl.sum(mt.__final_block_betas_bias_corrected**2)
                - hl.sum(mt.__final_block_betas_bias_corrected) ** 2 / n_blocks
            )
            / (n_blocks - 1)
            / n_blocks,
        ],
    )

    # convert coefficient to heritability estimate
    mt = mt.annotate_cols(
        phenotype=mt.__y_name,
        mean_chi_sq=hl.agg.mean(mt.__y),
        intercept=hl.struct(estimate=mt.__final_betas[0], standard_error=hl.sqrt(mt.__final_jackknife_variance[0])),
        snp_heritability=hl.struct(
            estimate=(M / hl.agg.mean(mt.__n)) * mt.__final_betas[1],
            standard_error=hl.sqrt((M / hl.agg.mean(mt.__n)) ** 2 * mt.__final_jackknife_variance[1]),
        ),
    )

    # format and return results
    ht = mt.cols()
    ht = ht.key_by(ht.phenotype)
    ht = ht.select(ht.mean_chi_sq, ht.intercept, ht.snp_heritability)

    ht_tmp_file = new_temp_file()
    ht.write(ht_tmp_file)
    ht = hl.read_table(ht_tmp_file)

    return ht
