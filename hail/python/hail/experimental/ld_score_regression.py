
import hail as hl
from hail.expr.expressions import *
from hail.typecheck import *
from hail.table import Table
from hail.linalg import BlockMatrix
from hail.matrixtable import MatrixTable
from hail.utils import wrap_to_list, new_temp_file

@typecheck(weight_expr=expr_float64,
           ld_score_expr=expr_numeric,
           chi_squared_exprs=oneof(expr_float64,
                                   sequenceof(expr_float64)),
           n_samples_exprs=oneof(expr_numeric,
                                 sequenceof(expr_numeric)),
           n_blocks=int,
           two_step_threshold=int,
           n_reference_panel_variants=nullable(int))
def ld_score_regression(weight_expr,
                        ld_score_expr,
                        chi_squared_exprs,
                        n_samples_exprs,
                        n_blocks=200,
                        two_step_threshold=30,
                        n_reference_panel_variants=None) -> Table:
    """Given a set or multiple sets of genome-wide association study (GWAS) summary
    statistics, :func:`.ld_score_regression` estimates the heritability of a trait
    or set of traits and the level of confounding biases present in the underlying 
    studies by regressing chi-squared statistics on LD scores, leveraging the model:

    .. math::

        E[\\chi_j^2] = 1 + Na + \\frac{Nh_g^2}{M}l_j

    *  :math:`E[\\chi_j^2]` is the expected chi-squared statistic resulting from a test
       of association between variant :math:`j` and a trait.
    *  :math:`l_j = \\sum_{k} r_{jk}^2` is the LD score of variant :math:`j`, calculated
       as the sum of squared correlation coefficients between variant `j` and variants in a set
       :math:`k` in a window around variant :math:`j`. See :func:`ld_score` for further details.
    *  :math:`a` is a measure of the contribution of confounding biases, such as sample
       relatedness and uncontrolled population structure, to the association test statistic.
    *  :math:`h_g^2` is the SNP-heritability, or the proportion of variation in the trait
       explained by the effects of variants included in the regression model above.
    *  :math:`M` is the number of variants over which :math:`h_g^2` is being estimated.
    *  :math:`N` is the number of samples in the underlying association study.

    For more details on the method implemented in this function, see:

    * `LD Score regression distinguishes confounding from polygenicity in genome-wide association studies (Bulik-Sullivan et al, 2015) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4495769/>`__

    Examples
    --------

    >>> mt_gwas = hl.read_matrix_table('data/univariate_ld_score_regression.chr22_sample.mt')
    >>> ht_results = hl.experimental.ld_score_regression(
    ...     weight_expr=mt_gwas['ld_score'],
    ...     ld_score_expr=mt_gwas['ld_score'],
    ...     chi_squared_exprs=mt_gwas['chi_squared'],
    ...     n_samples_exprs=mt_gwas['n_complete_samples'])

    >>> ht_gwas = hl.read_table('data/univariate_ld_score_regression.chr22_sample.ht')
    >>> ht_results = hl.experimental.ld_score_regression(
    ...     weight_expr=ht_gwas['ld_score'],
    ...     ld_score_expr=ht_gwas['ld_score'],
    ...     chi_squared_exprs=ht_gwas['50_irnt_chi_squared'],
    ...     n_samples_exprs=ht_gwas['50_irnt_n_complete_samples'])

    >>> ht_gwas = hl.read_table('data/univariate_ld_score_regression.chr22_sample.ht')
    >>> ht_results = hl.experimental.ld_score_regression(
    ...     weight_expr=ht_gwas['ld_score'],
    ...     ld_score_expr=ht_gwas['ld_score'],
    ...     chi_squared_exprs=[ht_gwas['50_irnt_chi_squared'],
    ...                        ht_gwas['20160_chi_squared']],
    ...     n_samples_exprs=[ht_gwas['50_irnt_n_complete_samples'],
    ...                      ht_gwas['20160_n_complete_samples']])


    This function returns a :class:`Table` with one row per set of summary statistics passed to 
    the ``chi_squared_exprs`` argument. The following row-indexed fields are included in the table:

    *  **phenotype** (:py:data:`.tstr`) -- The name of the phenotype. The returned table is
       keyed by this field. See the notes below for details on the possible values of this field. 
    *  **mean_chi_squared** (:py:data:`.tfloat64`) -- The mean chi-squared test statistic for
       the given phenotype.
    *  **intercept** (`Struct`) -- Contains fields:

       -  **estimate** (:py:data:`.tfloat64`) -- A point estimate of the intercept
          :math:`1 + Na` in the LD score regression model.
       -  **standard_error**  (:py:data:`.tfloat64`) -- An estimated standard error of the
          intercept point estimate.

    *  **snp_heritability** (`Struct`) -- Contains fields:

       -  **estimate** (:py:data:`.tfloat64`) -- A point estimate of the SNP-heritability 
          :math:`h_g^2`.
       -  **standard_error** (:py:data:`.tfloat64`) -- An estimated standard error of the
          :math:`h_g^2` point estimate.

    Notes
    -----
    The ``exprs`` provided as arguments to :func:`.ld_score_regression` can originate
    from either a :class:`Table` or a :class:`MatrixTable` (though they all must
    originate from the same object).

    **If the arguments originate from a table, then:**

    *  The table is assumed to be keyed by fields ``locus`` of type :class:`.tlocus` and
       ``alleles``, a :py:data:`.tarray` of :py:data:`.tstr` elements.
    *  ``weight_expr``, ``ld_score_expr``, ``chi_squared_exprs``, and ``n_samples_exprs``
       are assumed to be row-indexed fields of the table. 
    *  The number of expressions passed to ``n_samples_exprs`` must be equal to either one or
       the number of expressions passed to ``chi_squared_exprs``. If just one expression is
       passed to ``n_samples_exprs``, that sample size expression is assumed to apply to all
       sets of statistics passed to ``chi_squared_exprs``. Otherwise, the first expression
       passed to ``n_samples_exprs`` will correspond to the first expression passed to
       ``chi_squared_exprs``, the second expression passed to ``n_samples_exprs`` will correspond
       to the second expression passed to ``chi_squared_exprs``, etc.
    *  The ``phenotype`` field that keys the table returned by :func:`.ld_score_regression`
       will have generic values ``y0``, ``y1``, etc. corresponding to the ``0th``, ``1st``, etc.
       expressions passed to the ``chi_squared_exprs`` argument.
    
    **If the arguments originate from a matrix table, then:**

    *  The dimensions of the matrix table are assumed to be variants (rows)
       by phenotypes (columns).
    *  The rows of the matrix table are assumed to be keyed by fields ``locus`` of type
       :class:`.tlocus` and ``alleles``, a :py:data:`.tarray` of :py:data:`.tstr` elements.
    *  The columns of the matrix table are assumed to be keyed by a field of type
       :py:data:`.tstr` that uniquely identifies phenotypes represented in the matrix
       table. The column key must be a single expression; compound keys are not accepted.
    *  ``weight_expr``, and ``ld_score_expr`` are assumed to be row-indexed fields.
    *  ``chi_squared_exprs`` is assumed to be a single entry-indexed field (not a list of
       fields).
    *  ``n_samples_exprs`` is assumed to be a single entry-indexed field (not a list of
       fields).
    *  The ``phenotype`` field that keys the table returned by :func:`.ld_score_regression`
       will have values corresponding to the column keys of the input matrix table.

    Note
    ----
    Chi-squared statistics can be derived from the :math:`t` or :math:`Z` statistics
    typically provided in GWAS summary statistics files.

    A :math:`t`-distribution approaches a standard normal distribution for moderate
    sample sizes (:math:`n > 30`), such as those found in most GWAS. 

    Further, if a random variable :math:`Z` follows a standard normal distribution, 
    then :math:`Z^2` follows a :math:`\\chi_{df=1}^2` distribution. 

    In short, chi-squared statistics can generally be calculated from GWAS summary 
    statistics as :math:`X^2 = t^2` or :math:`X^2 = Z^2`, given :math:`t` or :math:`Z` 
    test statistics.

    Warning
    -------
    :func:`.ld_score_regression` considers only the rows for which both row fields 
    ``weight_expr`` and ``ld_score_expr`` are defined. Rows with missing values in either
    of these fields are removed prior to fitting the LD score regression model.

    Parameters
    ----------
    weight_expr : :class:`.Float64Expression`
                  Row-indexed expression for the LD scores used to derive variant weights in
                  the model.
    ld_score_expr : :class:`.Float64Expression`
                    Row-indexed expression for the LD scores used as covariates in the model.
    chi_squared_exprs : :class:`.Float64Expression` or :obj:`list` of :class:`.Float64Expression`
                        One or more row-indexed (if originating from a table) or entry-indexed
                        (if originating from a matrix table) expressions for chi-squared statistics
                        resulting from genome-wide association studies.
    n_samples_exprs: :class:`.NumericExpression` or :obj:`list` of :class:`.NumericExpression`
                     One or more row-indexed (if originating from a table) or entry-indexed
                     (if originating from a matrix table) expressions indicating the number
                     of samples used in the studies that generated the test statistics
                     supplied to the ``chi_squared_exprs`` argument.
    n_blocks : :obj:`int`
               The number of blocks used in the jackknife approach to estimating standard errors.
    two_step_threshold : :obj:`int`
                         In the two-step procedure used to fit the LD score regression model,
                         variants with chi-squared statistics greater than ``two_step_threshold``
                         are excluded in the estimation of the model intercept that occurs in the
                         first step.
    n_reference_panel_variants : :obj:`int`, optional
                                 Number of variants over which SNP-heritability :math:`h_g^2` is estimated. 
                                 If not supplied, assumed to be the number of variants in the LD score 
                                 regression.
    """

    chi_squared_exprs = wrap_to_list(chi_squared_exprs)
    n_samples_exprs = wrap_to_list(n_samples_exprs)

    assert (len(chi_squared_exprs) == len(n_samples_exprs)) or (len(n_samples_exprs) == 1)
    __p = 2  # number of covariates, including intercept -- will be larger (potentially 1000s) in partitioned version

    ds = chi_squared_exprs[0]._indices.source

    analyze('ld_score_regression/weight_expr', weight_expr, ds._row_indices)
    analyze('ld_score_regression/ld_score_expr', ld_score_expr, ds._row_indices)

    if isinstance(ds, MatrixTable):
        assert len(chi_squared_exprs) == 1
        assert len(n_samples_exprs) == 1
        col_key = list(ds.col_key)
        assert len(col_key) == 1

        analyze('ld_score_regression/chi_squared_expr', chi_squared_exprs[0], ds._entry_indices)
        analyze('ld_score_regression/n_samples_expr', n_samples_exprs[0], ds._entry_indices)

        ds = ds._select_all(row_exprs={'__locus': ds['locus'],
                                       '__alleles': ds['alleles'],
                                       '__w_initial': weight_expr,
                                       '__w_initial_floor': hl.max(weight_expr, 1.0),
                                       '__x': ld_score_expr,
                                       '__x_floor': hl.max(ld_score_expr, 1.0)},
                            row_key=['__locus', '__alleles'],
                            col_exprs={'__y_name': ds[col_key[0]]},
                            col_key=['__y_name'],
                            entry_exprs={'__y': chi_squared_exprs[0],
                                         '__n': n_samples_exprs[0]})
        ds = ds.annotate_entries(**{'__w': ds['__w_initial']})

        ds = ds.filter_rows(hl.is_defined(ds['__locus']) &
                            hl.is_defined(ds['__alleles']) &
                            hl.is_defined(ds['__w_initial']) &
                            hl.is_defined(ds['__x']))

    else:
        assert isinstance(ds, Table)
        for y in chi_squared_exprs:
            analyze('ld_score_regression/chi_squared_expr', y, ds._row_indices)
        for n in n_samples_exprs:
            analyze('ld_score_regression/n_samples_expr', n, ds._row_indices)

        ys = ['__y{:}'.format(i) for i, _ in enumerate(chi_squared_exprs)]
        ws = ['__w{:}'.format(i) for i, _ in enumerate(chi_squared_exprs)]
        ns = ['__n{:}'.format(i) for i, _ in enumerate(n_samples_exprs)]

        ds = ds.select(**dict(**{'__locus': ds['locus'],
                                 '__alleles': ds['alleles'],
                                 '__w_initial': weight_expr,
                                 '__x': ld_score_expr},
                              **{y: chi_squared_exprs[i] for i, y in enumerate(ys)},
                              **{w: weight_expr for w in ws},
                              **{n: n_samples_exprs[i] for i, n in enumerate(ns)}))
        ds = ds.key_by(ds['__locus'], ds['__alleles'])
        ds = ds.filter(hl.is_defined(ds['__locus']) &
                       hl.is_defined(ds['__alleles']) &
                       hl.is_defined(ds['__w_initial']) &
                       hl.is_defined(ds['__x']))

        table_tmp_file = new_temp_file()
        ds.write(table_tmp_file, overwrite=True)
        ds = hl.read_table(table_tmp_file)

        hts = [ds.select(**{'__w_initial': ds['__w_initial'],
                            '__w_initial_floor': hl.max(ds['__w_initial'], 1.0),
                            '__x': ds['__x'],
                            '__x_floor': hl.max(ds['__x'], 1.0),
                            '__y_name': y.strip('_'),
                            '__y': ds[ys[i]],
                            '__w': ds[ws[i]],
                            '__n': hl.int(ds[ns[i]])}) for i, y in enumerate(ys)]
        mts = [ht.to_matrix_table(row_key=['__locus', '__alleles'], 
                                  col_key=['__y_name'],
                                  row_fields=['__w_initial', '__w_initial_floor', '__x', '__x_floor']) for ht in hts]
        ds = mts[0]
        for i in range(1, len(mts)):
            ds = ds.union_cols(mts[i])

    ds_tmp_file = new_temp_file()
    ds.write(ds_tmp_file, overwrite=True)
    mt = hl.read_matrix_table(ds_tmp_file)

    mt = mt._annotate_all(col_exprs={'__initial_betas': [1.0, (hl.agg.mean(mt['__y']) - 1.0)/hl.agg.mean(mt['__x'])]},
                          row_exprs={'__idx': hl.scan.count()},
                          entry_exprs={'__in_step1': mt['__y'] < two_step_threshold})
    mt = mt.annotate_cols(__step1_betas=mt['__initial_betas'])

    for i in range(3):
        mt = mt.annotate_entries(__w=1.0/(mt['__w_initial_floor'] * 
                                          2.0 * (mt['__step1_betas'][0] + mt['__step1_betas'][1] * mt['__x_floor'])**2))
        mt = mt.annotate_cols(__step1_betas=hl.agg.filter(mt['__in_step1'],
                                                          hl.agg.linreg(y=mt['__y'],
                                                                        x=[1.0, mt['__x']],
                                                                        weight=mt['__w']).beta))
    variants_per_block = mt.count_rows()/n_blocks
    mt = mt.annotate_rows(__block=hl.int(mt['__idx']/variants_per_block))

    mt = mt.annotate_cols(__step1_block_betas=[
        hl.agg.filter((mt['__block'] != i) & mt['__in_step1'],
                      hl.agg.linreg(y=mt['__y'],
                                    x=[1.0, mt['__x']],
                                    weight=mt['__w']).beta) for i in range(n_blocks)])

    mt = mt.annotate_cols(__step1_block_pseudovalues=hl.map(lambda x: n_blocks * mt['__step1_betas'] - (n_blocks - 1) * x,
                                                            mt['__step1_block_betas']))

    mt_step1_tmp = new_temp_file()
    mt.write(mt_step1_tmp, overwrite=True)
    mt = hl.read_matrix_table(mt_step1_tmp)

    mt = mt.annotate_cols(__step1_jackknife_mean=hl.map(lambda i: hl.mean(hl.map(lambda x: x[i], 
                                                                                 mt['__step1_block_pseudovalues'])),
                                                        hl.range(0, __p)))
    mt = mt.annotate_cols(__step1_jackknife_variance=hl.map(lambda i: (hl.sum(hl.map(lambda x: x[i]**2/(n_blocks - 1),
                                                                                     mt['__step1_block_pseudovalues'])) - 
                                                                       mt['__step1_jackknife_mean'][i]**2)/n_blocks,
                                                            hl.range(0, __p)))

    mt = mt._annotate_all(col_exprs={'__step2_betas': mt['__initial_betas']},
                          entry_exprs={'__in_step2': hl.is_defined(mt['__y'])})

    for i in range(3):
        mt = mt.annotate_entries(__w=1.0/(mt['__w_initial_floor'] * 
                                          2.0 * (mt['__step2_betas'][0] + mt['__step2_betas'][1] * mt['__x_floor'])**2))
        mt = mt.annotate_cols(__step2_betas=[mt['__step1_betas'][0],
                                             hl.agg.filter(mt['__in_step2'],
                                                           hl.agg.linreg(y=mt['__y'] - mt['__step1_betas'][0],
                                                                         x=[mt['__x']],
                                                                         weight=mt['__w']).beta[0])])

    mt = mt.annotate_cols(__step2_block_betas=[
        hl.agg.filter((mt['__block'] != i) & mt['__in_step2'],
                      hl.agg.linreg(y=mt['__y'] - mt['__step1_betas'][0],
                                    x=[mt['__x']],
                                    weight=mt['__w']).beta[0]) for i in range(n_blocks)])

    mt = mt.annotate_cols(__step2_block_pseudovalues=hl.map(lambda x: n_blocks * mt['__step2_betas'][1] - (n_blocks - 1) * x,
                                                            mt['__step2_block_betas']))

    mt_step2_tmp = new_temp_file()
    mt.write(mt_step2_tmp, overwrite=True)
    mt = hl.read_matrix_table(mt_step2_tmp)

    mt = mt.annotate_cols(__step2_jackknife_mean=hl.mean(mt['__step2_block_pseudovalues']))
    mt = mt.annotate_cols(__step2_jackknife_variance=(hl.sum(mt['__step2_block_pseudovalues']**2)/(n_blocks - 1) - 
                                                      mt['__step2_jackknife_mean']**2)/n_blocks)

    mt = mt.annotate_entries(__step2_initial_w=1.0/(mt['__w_initial_floor'] * 
                                                    2.0 * (mt['__initial_betas'][0] + mt['__initial_betas'][1] * mt['__x_floor'])**2))
    
    mt = mt.annotate_cols(__final_betas=[mt['__step1_betas'][0], mt['__step2_betas'][1]],
                          __c=hl.agg.sum(mt['__step2_initial_w'] * mt['__x'])/hl.agg.sum(mt['__step2_initial_w'] * mt['__x']**2))    

    mt = mt.annotate_cols(__final_block_betas=hl.map(lambda i: [mt['__step1_block_betas'][i][0],
                                                                mt['__step2_block_betas'][i] - mt['__c'] * (mt['__step1_block_betas'][i][0] - 
                                                                                                            mt['__final_betas'][0])], 
                                                 hl.range(0, n_blocks)))
    
    mt = mt.annotate_cols(__final_block_pseudovalues=hl.map(lambda x: n_blocks * mt['__final_betas'] - (n_blocks - 1) * x,
                                                            mt['__final_block_betas']))

    mt = mt.annotate_cols(__final_jackknife_mean=hl.map(lambda i: hl.mean(hl.map(lambda x: x[i],
                                                                                 mt['__final_block_pseudovalues'])),
                                                        hl.range(0, __p)))
    mt = mt.annotate_cols(__final_jackknife_variance=hl.map(lambda i: (hl.sum(hl.map(lambda x: x[i]**2/(n_blocks - 1),
                                                                                     mt['__final_block_pseudovalues'])) - mt['__final_jackknife_mean'][i]**2)/n_blocks,
                                                            hl.range(0, __p)))
    
    mt_final_tmp = new_temp_file()
    mt.write(mt_final_tmp, overwrite=True)
    mt = hl.read_matrix_table(mt_final_tmp)

    if not n_reference_panel_variants:
        M = mt.count_rows()
    else:
        M = n_reference_panel_variants

    mt = mt.annotate_cols(phenotype=mt['__y_name'],
                          mean_chi_squared=hl.agg.mean(mt['__y']),
                          intercept=hl.struct(estimate=mt['__final_betas'][0],
                                              standard_error=hl.sqrt(mt['__final_jackknife_variance'][0])),
                          snp_heritability=hl.struct(estimate=(M/hl.agg.mean(mt['__n'])) * mt['__final_betas'][1],
                                                     standard_error=hl.sqrt((M/hl.agg.mean(mt['__n']))**2 * 
                                                                            mt['__final_jackknife_variance'][1])))


    ht = mt.cols()
    ht = ht.key_by(ht['phenotype'])
    ht = ht.select(ht['mean_chi_squared'], ht['intercept'], ht['snp_heritability'])

    return ht
