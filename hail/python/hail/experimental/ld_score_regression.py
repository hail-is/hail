
import hail as hl
from hail.typecheck import *
from hail.table import Table
from hail.linalg import BlockMatrix
from hail.matrixtable import MatrixTable
from hail.utils import wrap_to_list, new_temp_file

@typecheck(locus_expr=expr_locus(),
           weight_expr=expr_float64,
           chi_squared_exprs=oneof(expr_float64,
                                   sequenceof(expr_float64)),
           n_samples_exprs=oneof(expr_numeric,
                                 sequenceof(expr_numeric)),
           ld_score_expr=expr_numeric,
           n_iterations=int,
           n_blocks=int,
           two_step_threshold=int,
           n_reference_panel_snps=nullable(int))
def ld_score_regression(locus_expr,
                        weight_expr,
                        chi_squared_exprs,
                        n_samples_exprs,
                        ld_score_expr,
                        phenotype_expr=None,
                        n_iterations=2,
                        n_blocks=200,
                        two_step_threshold=30,
                        n_reference_panel_variants=None) -> Table:
    """Function to estimate trait heritability and level of confounding biases based on
    GWAS summary statistics.
 
    Given a set or multiple sets of GWAS summary statistics, :func:`.ld_score_regression`
    estimates the heritability of the trait or set of traits and the level of confounding
    biases present in the underlying studies using the following model:

    .. math::

        E[\\chi^2]

        
        will return
    a :class:`Table` with:
    1) A heritability estimate and associated standard error of the estimate for each
    set of summary statistics passed as an argument to `chi_squared_exprs`. This
    2) An 
    of the heritability of the phenotypes from which
    the summary statistics were derived and of the level of confounding biases

    Notes
    -----
    The `exprs` provided as arguments to :func:`.ld_score_regression` can originate
    from either a :class:`Table` or a :class:`MatrixTable`.

    If the `exprs` originate from a :class:`Table`, then:
    * `locus_expr`, `weight_expr`, `chi_squared_expr`, `n_samples_exprs`,
      and `ld_score_expr` are assumed to be row-indexed fields of the :class:`Table`. 
    * The `phenotype_expr` argument is not used.
    * The :class:`Table` returned by :func:`.ld_score_regression` is keyed by a column
    
    If the `exprs` originate from a :class:`MatrixTable`, then :func:`.ld_score_regression`
    assumes the following:
    * The dimensions of the :class:`MatrixTable` are assumed to be variants (rows)
    by phenotypes (columns).
    * `locus_expr`, `weight_expr`, and `ld_score_expr` are assumed to be
      row-indexed fields.
    * `chi_squared_exprs` is assumed to be a single row- and column-indexed
      entry field (not a list of fields).
    * `phenotype_expr` is assumed to be a column-indexed field. If the user does not
      supply an argument to `phenotype_expr`, :func:`.ld_score_regression` assumes the
      column key field of the :class:`MatrixTable` defines the unique phenotypes.

    Warning
    -------
    :func:`.ld_score_regression` considers only the rows for which row fields 
    `locus_expr`, `weight_expr`, and `ld_score_expr` are defined. Rows with missing
    values in these fields are removed prior to fitting the LD score regression
    model.

    Warning
    -------
    If a list of `exprs` is provided as an argument to `n_samples_exprs`, the length of the
    list must equal the length of the list passed to The order of expressions passed to the `n_samples_expr` argument must match the order of the
    expressions passed to the `chi_squared_exprs` argument. For example, if the argument
    `chi_squared_exprs=[ht.phenotype1_chi_squared, ht.phenotype2_chi_squared]` is passed to
    :func:`.ld_score_regression`, then the argument `n_samples_expr=[ht.n_samples_phenotype1, 
    ht.n_samples_phenotype2]` should also be passed to the function.

    """

    ds = locus_expr._indices.source
    ds_tmp_file = new_temp_file()

    chi_squared_exprs = wrap_to_list(chi_squared_exprs)
    n_samples_exprs = wrap_to_list(n_samples_exprs)

    assert (len(chi_squared_exprs) == len(n_samples_exprs)) or (len(n_samples_exprs) == 1)
    __p = 2  # number of covariates, including intercept -- will be larger (potentially 1000s) in partitioned version

    analyze('ld_score_regression/locus_expr', locus_expr, ds._row_indices)
    analyze('ld_score_regression/weight_expr', weight_expr, ds._row_indices)
    analyze('ld_score_regression/ld_score_expr', ld_score_expr, ds._row_indices)

    if isinstance(ds, MatrixTable):
        assert len(chi_squared_exprs) == 1
        assert len(n_samples_exprs) == 1
        if phenotype_expr is None:
            phenotype_expr = ds.col_key

        analyze('ld_score_regression/chi_squared_expr', chi_squared_exprs[0], ds._entry_indices)
        analyze('ld_score_regression/n_samples_expr', n_samples_exprs[0], ds._entry_indices)

        ds = ds._select_all(row_exprs={'__locus': locus_expr,
                                       '__w_initial': weight_expr,
                                       '__x': ld_score_expr},
                            row_key=['__locus'],
                            col_exprs={'__y_name': phenotype_expr},
                            col_key=['__y_name'],
                            entry_exprs={'__y': chi_squared_exprs[0],
                                         '__n': n_samples_exprs[0]})
        ds = ds.annotate_entries(**{'__w': ds['__w_initial']})

        ds = ds.filter_rows(hl.is_defined(ds['__locus']) &
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

        ds = ds.select(**dict(**{'__locus': locus_expr,
                                 '__w_initial': weight_expr,
                                 '__x': ld_score_expr},
                              **{y: chi_squared_exprs[i] for i, y in enumerate(ys)},
                              **{w: weight_expr for w in ws},
                              **{n: n_samples_exprs[i] for i, n in enumerate(ns)}))
        ds = ds.key_by(ds['__locus'])
        ds = ds.filter(hl.is_defined(ds['__locus']) &
                       hl.is_defined(ds['__w_initial']) &
                       hl.is_defined(ds['__x']))

        table_tmp_file = new_temp_file()
        ds.write(table_tmp_file, overwrite=True)
        ds = hl.read_table(table_tmp_file)

        hts = [ds.select(**{'__w_initial': ds['__w_initial'],
                            '__w_initial_floor': hl.max(ds['__w_initial'], 1.0),
                            '__x': ds['__x'],
                            '__x_floor': hl.max(ds['__x'], 1.0),
                            '__y_name': y,
                            '__y': ds[ys[i]],
                            '__w': ds[ws[i]],
                            '__n': ds[ns[i]]}) for i, y in enumerate(ys)]
        mts = [ht.to_matrix_table(row_key=['__locus'], 
                                  col_key=['__y_name'],
                                  row_fields=['__w_initial', '__w_initial_floor', '__x', '__x_floor']) for ht in hts]
        ds = MatrixTable.union_cols(*mts)

    ds.write(ds_tmp_file, overwrite=True)
    mt = hl.read_matrix_table(ds_tmp_file)

    mt = mt._annotate_all(col_exprs={'__initial_betas': [1.0, (hl.agg.mean(mt['__y']) - 1.0)/hl.agg.mean(mt['__x'])]},
                          row_exprs={'__idx': hl.scan.count(),
                                     '__x_floor': hl.max(mt['__x'], 1.0),
                                     '__w_initial_floor': hl.max(mt['__w_initial'], 1.0)},
                          entry_exprs={'__in_step1': mt['__y'] < two_step_threshold})
    mt = mt.annotate_cols(__step1_betas=mt['__initial_betas'])

    for i in range(n_iterations + 1):
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

    for i in range(n_iterations + 1):
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

    mt = mt.annotate_cols(mean_chi_squared=hl.agg.mean(mt['__y']),
                          intercept=hl.struct(estimate=mt['__final_betas'][0],
                                              standard_error=hl.sqrt(mt['__final_jackknife_variance'][0])),
                          observed_scale_heritability=hl.struct(estimate=(M/hl.agg.mean(mt['__n'])) * mt['__final_betas'][1],
                                                                standard_error=hl.sqrt((M/hl.agg.mean(mt['__n']))**2 * 
                                                                                       mt['__final_jackknife_variance'][1])))


    ht = mt.cols()
    ht = ht.select(ht['mean_chi_squared'], ht['intercept'], ht['observed_scale_heritability'])

    return ht
