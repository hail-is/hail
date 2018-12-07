
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
           ld_score_exprs=oneof(expr_numeric,
                                sequenceof(expr_numeric)),
           n_iterations=int,
           n_blocks=int,
           two_step_threshold=int,
           n_reference_panel_snps=nullable(int))
def ld_score_regression(locus_expr,
                        weight_expr,
                        chi_squared_exprs,
                        n_samples_exprs,
                        ld_score_exprs,
                        phenotype_expr=None,
                        n_iterations=2,
                        n_blocks=200,
                        two_step_threshold=30,
                        n_reference_panel_variants=None) -> Table:
    """Run LD score regression.

    Warning
    -------
    :func:`.ld_score_regression` considers only the rows for which **all** row fields
    (locus, chi-squared test statistics, LD score annotations, LD score weight, number of samples)
    are defined. Rows with missing values are removed prior to fitting the LD
    score regression model.

    Warning
    -------
    The order of expressions passed to the `n_samples_expr` argument must match the order of the
    expressions passed to the `chi_squared_exprs` argument. For example, if the argument
    `chi_squared_exprs=[ht.phenotype1_chi_squared, ht.phenotype2_chi_squared]` is passed to
    :func:`.ld_score_regression`, then the argument `n_samples_expr=[ht.n_samples_phenotype1, 
    ht.n_samples_phenotype2]` should also be passed to the function.

    """

    ds = locus_expr._indices.source
    ds_tmp_file = new_temp_file()

    __p = 2
    chi_squared_exprs = wrap_to_list(chi_squared_exprs)
    n_samples_exprs = wrap_to_list(n_samples_exprs)
    assert (len(chi_squared_exprs) == len(n_samples_exprs)) or (len(n_samples_exprs) == 1)

    analyze('ld_score_regression/locus_expr', locus_expr, ds._row_indices)
    analyze('ld_score_regression/weight_expr', weight_expr, ds._row_indices)
    analyze('ld_score_regression/ld_score_expr', ld_score_expr, ds._row_indices)

    if is_instance(ds, MatrixTable):
        assert len(chi_squared_exprs) == 1
        assert len(n_samples_exprs) == 1
        if phenotype_expr is None:
            phenotype_expr = ds.col_key

        analyze('ld_score_regression/chi_squared_expr', chi_squared_exprs[0], ds._entry_indices)
        analyze('ld_score_regression/n_samples_expr', n_samples_exprs[0], ds._entry_indices)

        ds = ds._select_all(row_exprs={'__locus': locus_expr,
                                       '__x': ld_score_expr,
                                       '__w_initial': weight_expr},
                            row_key=['__locus'],
                            col_exprs={'__y_name': phenotype_expr},
                            col_key=['__y_name'],
                            entry_exprs={'__y': chi_squared_exprs[0],
                                         '__n': n_samples_exprs[0]})
        ds = ds.annotate_entries(**{'__w': ds['__w_initial']})

        ds = ds.filter_rows(hl.is_defined(ds['__locus']) &
                            hl.is_defined(ds['__x']) &
                            hl.is_defined(ds['__w_initial']))

    else:
        assert isinstance(ds, Table)
        for y in chi_squared_exprs:
            analyze('ld_score_regression/chi_squared_expr', y, ds._row_indices)
        for n in n_samples_exprs:
            analyze('ld_score_regression/n_samples_expr', n, ds._row_indices)

        ys = ['__y{:}'.format(i) for i in range(len(chi_squared_exprs))]
        ws = ['__w{:}'.format(i) for i in range(len(chi_squared_exprs))]
        ns = ['__n{:}'.format(i) for i in range(len(n_samples_exprs))]
        
        ds = ds.select(**dict(**{'__locus': locus_expr,
                                 '__x': ld_score_expr,
                                 '__w_initial': weight_expr},
                              **{y: chi_squared_exprs[i] for i, y in enumerate(ys)},
                              **{w: weight_expr for w in ws},
                              **{n: n_samples_exprs[i] for i, n in enumerate(ns)}))
        ds = ds.key_by(ds['__locus'])
        ds = ds.filter(hl.is_defined(ds['__locus']) & 
                       hl.is_defined(ds['__x']) & 
                       hl.is_defined(ds['__w_initial']))

        table_tmp_file = new_temp_file()
        ds.write(table_tmp_file, overwrite=True)
        ds = hl.read_table(table_tmp_file)

        hts = [ds.select(__x=ds['__x'],
                         __w_initial=ds['__w_initial'], 
                         __y_name=y, 
                         __y=ds[ys[i]], 
                         __w=ds[ws[i]], 
                         __n=ds[ns[i]]) for i, y in enumerate(ys)]
        mts = [ht.to_matrix_table(row_key=['__locus'], 
                                  col_key=['__y_name'],
                                  row_fields=['__x', '__w_initial']) for ht in hts]
        ds = MatrixTable.union_cols(*mts)

    ds.write(ds_tmp_file, overwrite=True)
    mt = hl.read_matrix_table(ds_tmp_file)

    mt = mt._annotate_all(col_exprs={'__initial_betas': [1.0, (hl.agg.mean(mt['__y']) - 1.0)/hl.agg.mean(mt['__x'])]},
                          row_exprs={'__idx': hl.scan.count(),
                                     '__x_floor': hl.max(mt['__x'], 1.0),
                                     '__w_initial_floor': hl.max(mt['__w_initial'], 1.0)},
                          entry_exprs={'__in_step1': mt['__y'] < 30})
    mt = mt._annotate_all(col_exprs={'__step1_betas': mt['__initial_betas']},
                          row_exprs={'__block': hl.int(mt['__idx']/(mt.count_rows()/n_blocks))})

    for i in range(n_iterations + 1):
        mt = mt.annotate_entries(__w=1.0/(mt['__w_initial_floor'] * 2.0 * (mt['__step1_betas'][0] + mt['__step1_betas'][1] * mt['__x_floor'])**2))
        mt = mt.annotate_cols(__step1_betas=hl.agg.filter(mt['__in_step1'],
                                                          hl.agg.linreg(y=mt['__y'],
                                                                        x=[1.0, mt['__x']],
                                                                        weight=mt['__w']).beta))

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

    mt = mt.annotate_cols(**{'__step1_jackknife_mean':[hl.mean(hl.map(lambda x: x[i], 
                                                                  mt['__step1_block_pseudovalues'])) for i in range(__p)],
                             '__step1_jackknife_variance':[(hl.sum((hl.map(lambda x: x[i],
                                                                           mt['__step1_block_pseudovalues']) - 
                                                                    hl.mean(hl.map(lambda x: x[i],
                                                                                   mt['__step1_block_pseudovalues'])))**2)/(n_blocks - 1))/n_blocks for i in range(__p)]})





