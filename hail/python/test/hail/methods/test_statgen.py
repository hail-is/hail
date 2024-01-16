import math
import os
import unittest

import numpy as np
import pytest

import hail as hl
import hail.expr.aggregators as agg
from hail import utils
from hail.linalg import BlockMatrix
from hail.utils import FatalError, new_temp_file
from hail.utils.java import Env, choose_backend

from ..helpers import fails_service_backend, qobtest, resource, skip_when_service_backend, test_timeout


class Tests(unittest.TestCase):
    @qobtest
    @pytest.mark.skipif('HAIL_TEST_SKIP_PLINK' in os.environ, reason='Skipping tests requiring plink')
    @fails_service_backend()
    def test_impute_sex_same_as_plink(self):
        ds = hl.import_vcf(resource('x-chromosome.vcf'))

        sex = hl.impute_sex(ds.GT, include_par=True)

        vcf_file = utils.uri_path(utils.new_temp_file(prefix="plink", extension="vcf"))
        out_file = utils.uri_path(utils.new_temp_file(prefix="plink"))

        hl.export_vcf(ds, vcf_file)

        utils.run_command(["plink", "--vcf", vcf_file, "--const-fid", "--check-sex", "--silent", "--out", out_file])

        plink_sex = hl.import_table(
            out_file + '.sexcheck', delimiter=' +', types={'SNPSEX': hl.tint32, 'F': hl.tfloat64}
        )
        plink_sex = plink_sex.select('IID', 'SNPSEX', 'F')
        plink_sex = plink_sex.select(
            s=plink_sex.IID,
            is_female=hl.if_else(
                plink_sex.SNPSEX == 2, True, hl.if_else(plink_sex.SNPSEX == 1, False, hl.missing(hl.tbool))
            ),
            f_stat=plink_sex.F,
        ).key_by('s')

        sex = sex.select('is_female', 'f_stat')

        self.assertTrue(plink_sex._same(sex.select_globals(), tolerance=1e-3))

        ds = ds.annotate_rows(aaf=(agg.call_stats(ds.GT, ds.alleles)).AF[1])

        self.assertTrue(hl.impute_sex(ds.GT)._same(hl.impute_sex(ds.GT, aaf='aaf')))

    backend_name = choose_backend()
    # Outside of Spark backend, "linear_regression_rows" just defers to the underscore nd version.
    linreg_functions = (
        [hl.linear_regression_rows, hl._linear_regression_rows_nd]
        if backend_name == "spark"
        else [hl.linear_regression_rows]
    )

    @qobtest
    @test_timeout(4 * 60)
    def test_linreg_basic(self):
        phenos = hl.import_table(resource('regressionLinear.pheno'), types={'Pheno': hl.tfloat64}, key='Sample')
        covs = hl.import_table(
            resource('regressionLinear.cov'), types={'Cov1': hl.tfloat64, 'Cov2': hl.tfloat64}, key='Sample'
        )

        mt = hl.import_vcf(resource('regressionLinear.vcf'))
        mt = mt.annotate_cols(pheno=phenos[mt.s].Pheno, cov=covs[mt.s])
        mt = mt.annotate_entries(x=mt.GT.n_alt_alleles()).cache()

        for linreg_function in self.linreg_functions:
            t1 = linreg_function(
                y=mt.pheno, x=mt.GT.n_alt_alleles(), covariates=[1.0, mt.cov.Cov1, mt.cov.Cov2 + 1 - 1]
            )
            t1 = t1.select(p=t1.p_value)

            t2 = linreg_function(y=mt.pheno, x=mt.x, covariates=[1.0, mt.cov.Cov1, mt.cov.Cov2])
            t2 = t2.select(p=t2.p_value)

            t3 = linreg_function(y=[mt.pheno], x=mt.x, covariates=[1.0, mt.cov.Cov1, mt.cov.Cov2])
            t3 = t3.select(p=t3.p_value[0])

            t4 = linreg_function(y=[mt.pheno, mt.pheno], x=mt.x, covariates=[1.0, mt.cov.Cov1, mt.cov.Cov2])
            t4a = t4.select(p=t4.p_value[0])
            t4b = t4.select(p=t4.p_value[1])

            self.assertTrue(t1._same(t2))
            self.assertTrue(t1._same(t3))
            self.assertTrue(t1._same(t4a))
            self.assertTrue(t1._same(t4b))

    def test_linreg_pass_through(self):
        phenos = hl.import_table(resource('regressionLinear.pheno'), types={'Pheno': hl.tfloat64}, key='Sample')

        mt = hl.import_vcf(resource('regressionLinear.vcf')).annotate_rows(foo=hl.struct(bar=hl.rand_norm(0, 1)))

        for linreg_function in self.linreg_functions:
            # single group
            lr_result = linreg_function(
                phenos[mt.s].Pheno, mt.GT.n_alt_alleles(), [1.0], pass_through=['filters', mt.foo.bar, mt.qual]
            )

            assert mt.aggregate_rows(hl.agg.all(mt.foo.bar == lr_result[mt.row_key].bar))

            # chained
            lr_result = linreg_function(
                [[phenos[mt.s].Pheno]], mt.GT.n_alt_alleles(), [1.0], pass_through=['filters', mt.foo.bar, mt.qual]
            )

            assert mt.aggregate_rows(hl.agg.all(mt.foo.bar == lr_result[mt.row_key].bar))

            # check types
            assert 'filters' in lr_result.row
            assert lr_result.filters.dtype == mt.filters.dtype

            assert 'bar' in lr_result.row
            assert lr_result.bar.dtype == mt.foo.bar.dtype

            assert 'qual' in lr_result.row
            assert lr_result.qual.dtype == mt.qual.dtype

            # should run successfully with key fields
            linreg_function([[phenos[mt.s].Pheno]], mt.GT.n_alt_alleles(), [1.0], pass_through=['locus', 'alleles'])

            # complex expression
            with pytest.raises(ValueError):
                linreg_function(
                    [[phenos[mt.s].Pheno]], mt.GT.n_alt_alleles(), [1.0], pass_through=[mt.filters.length()]
                )

    @test_timeout(local=3 * 60)
    def test_linreg_chained(self):
        phenos = hl.import_table(resource('regressionLinear.pheno'), types={'Pheno': hl.tfloat64}, key='Sample')
        covs = hl.import_table(
            resource('regressionLinear.cov'), types={'Cov1': hl.tfloat64, 'Cov2': hl.tfloat64}, key='Sample'
        )

        mt = hl.import_vcf(resource('regressionLinear.vcf'))
        mt = mt.annotate_cols(pheno=phenos[mt.s].Pheno, cov=covs[mt.s])
        mt = mt.annotate_entries(x=mt.GT.n_alt_alleles()).cache()

        for linreg_function in self.linreg_functions:
            t1 = linreg_function(y=[[mt.pheno], [mt.pheno]], x=mt.x, covariates=[1, mt.cov.Cov1, mt.cov.Cov2])

            def all_eq(*args):
                pred = True
                for a in args:
                    if (
                        isinstance(a, hl.expr.Expression)
                        and isinstance(a.dtype, hl.tarray)
                        and isinstance(a.dtype.element_type, hl.tarray)
                    ):
                        pred = pred & (
                            hl.all(
                                lambda x: x,
                                hl.map(
                                    lambda elt: ((hl.is_nan(elt[0]) & hl.is_nan(elt[1])) | (elt[0] == elt[1])),
                                    hl.zip(a[0], a[1]),
                                ),
                            )
                        )
                    else:
                        pred = pred & ((hl.is_nan(a[0]) & hl.is_nan(a[1])) | (a[0] == a[1]))
                return pred

            assert t1.aggregate(
                hl.agg.all(all_eq(t1.n, t1.sum_x, t1.y_transpose_x, t1.beta, t1.standard_error, t1.t_stat, t1.p_value))
            )

            mt2 = mt.filter_cols(mt.cov.Cov2 >= 0)
            mt3 = mt.filter_cols(mt.cov.Cov2 <= 0)

            # test that chained linear regression can replicate separate calls with different missingness
            t2 = hl.linear_regression_rows(y=mt2.pheno, x=mt2.x, covariates=[1, mt2.cov.Cov1])
            t3 = hl.linear_regression_rows(y=mt3.pheno, x=mt3.x, covariates=[1, mt3.cov.Cov1])

            chained = hl.linear_regression_rows(
                y=[
                    [hl.case().when(mt.cov.Cov2 >= 0, mt.pheno).or_missing()],
                    [hl.case().when(mt.cov.Cov2 <= 0, mt.pheno).or_missing()],
                ],
                x=mt.x,
                covariates=[1, mt.cov.Cov1],
            )
            chained = chained.annotate(r0=t2[chained.key], r1=t3[chained.key])
            assert chained.aggregate(
                hl.agg.all(
                    all_eq(
                        [chained.n[0], chained.r0.n],
                        [chained.n[1], chained.r1.n],
                        [chained.sum_x[0], chained.r0.sum_x],
                        [chained.sum_x[1], chained.r1.sum_x],
                        [chained.y_transpose_x[0][0], chained.r0.y_transpose_x],
                        [chained.y_transpose_x[1][0], chained.r1.y_transpose_x],
                        [chained.beta[0][0], chained.r0.beta],
                        [chained.beta[1][0], chained.r1.beta],
                        [chained.standard_error[0][0], chained.r0.standard_error],
                        [chained.standard_error[1][0], chained.r1.standard_error],
                        [chained.t_stat[0][0], chained.r0.t_stat],
                        [chained.t_stat[1][0], chained.r1.t_stat],
                        [chained.p_value[0][0], chained.r0.p_value],
                        [chained.p_value[1][0], chained.r1.p_value],
                    )
                )
            )

            # test differential missingness against each other
            phenos = [
                hl.case().when(mt.cov.Cov2 >= -1, mt.pheno).or_missing(),
                hl.case().when(mt.cov.Cov2 <= 1, mt.pheno).or_missing(),
            ]
            t4 = hl.linear_regression_rows(phenos, mt.x, covariates=[1])
            t5 = hl.linear_regression_rows([phenos], mt.x, covariates=[1])

            t5 = t5.annotate(**{
                x: t5[x][0] for x in ['n', 'sum_x', 'y_transpose_x', 'beta', 'standard_error', 't_stat', 'p_value']
            })
            assert t4._same(t5)

    def test_linear_regression_without_intercept(self):
        for linreg_function in self.linreg_functions:
            pheno = hl.import_table(
                resource('regressionLinear.pheno'), key='Sample', missing='0', types={'Pheno': hl.tfloat}
            )
            mt = hl.import_vcf(resource('regressionLinear.vcf'))
            ht = linreg_function(y=pheno[mt.s].Pheno, x=mt.GT.n_alt_alleles(), covariates=[])
            results = dict(hl.tuple([ht.locus.position, ht.row]).collect())
            self.assertAlmostEqual(results[1].beta, 1.5, places=6)
            self.assertAlmostEqual(results[1].standard_error, 1.161895, places=6)
            self.assertAlmostEqual(results[1].t_stat, 1.290994, places=6)
            self.assertAlmostEqual(results[1].p_value, 0.25317, places=6)

    # comparing to R:
    # y = c(1, 1, 2, 2, 2, 2)
    # x = c(0, 1, 0, 0, 0, 1)
    # c1 = c(0, 2, 1, -2, -2, 4)
    # c2 = c(-1, 3, 5, 0, -4, 3)
    # df = data.frame(y, x, c1, c2)
    # fit <- lm(y ~ x + c1 + c2, data=df)
    # summary(fit)["coefficients"]
    @pytest.mark.unchecked_allocator
    def test_linear_regression_with_cov(self):
        covariates = hl.import_table(
            resource('regressionLinear.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        pheno = hl.import_table(
            resource('regressionLinear.pheno'), key='Sample', missing='0', types={'Pheno': hl.tfloat}
        )

        mt = hl.import_vcf(resource('regressionLinear.vcf'))

        for linreg_function in self.linreg_functions:
            ht = linreg_function(
                y=pheno[mt.s].Pheno, x=mt.GT.n_alt_alleles(), covariates=[1.0, *list(covariates[mt.s].values())]
            )

            results = dict(hl.tuple([ht.locus.position, ht.row]).collect())

            self.assertAlmostEqual(results[1].beta, -0.28589421, places=6)
            self.assertAlmostEqual(results[1].standard_error, 1.2739153, places=6)
            self.assertAlmostEqual(results[1].t_stat, -0.22442167, places=6)
            self.assertAlmostEqual(results[1].p_value, 0.84327106, places=6)

            self.assertAlmostEqual(results[2].beta, -0.5417647, places=6)
            self.assertAlmostEqual(results[2].standard_error, 0.3350599, places=6)
            self.assertAlmostEqual(results[2].t_stat, -1.616919, places=6)
            self.assertAlmostEqual(results[2].p_value, 0.24728705, places=6)

            self.assertAlmostEqual(results[3].beta, 1.07367185, places=6)
            self.assertAlmostEqual(results[3].standard_error, 0.6764348, places=6)
            self.assertAlmostEqual(results[3].t_stat, 1.5872510, places=6)
            self.assertAlmostEqual(results[3].p_value, 0.2533675, places=6)

            self.assertTrue(np.isnan(results[6].standard_error))
            self.assertTrue(np.isnan(results[6].t_stat))
            self.assertTrue(np.isnan(results[6].p_value))

            self.assertTrue(np.isnan(results[7].standard_error))
            self.assertTrue(np.isnan(results[8].standard_error))
            self.assertTrue(np.isnan(results[9].standard_error))
            self.assertTrue(np.isnan(results[10].standard_error))

    def test_linear_regression_pl(self):
        covariates = hl.import_table(
            resource('regressionLinear.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        pheno = hl.import_table(
            resource('regressionLinear.pheno'), key='Sample', missing='0', types={'Pheno': hl.tfloat}
        )

        mt = hl.import_vcf(resource('regressionLinear.vcf'))

        for linreg_function in self.linreg_functions:
            ht = linreg_function(
                y=pheno[mt.s].Pheno, x=hl.pl_dosage(mt.PL), covariates=[1.0, *list(covariates[mt.s].values())]
            )

            results = dict(hl.tuple([ht.locus.position, ht.row]).collect())

            self.assertAlmostEqual(results[1].beta, -0.29166985, places=6)
            self.assertAlmostEqual(results[1].standard_error, 1.2996510, places=6)
            self.assertAlmostEqual(results[1].t_stat, -0.22442167, places=6)
            self.assertAlmostEqual(results[1].p_value, 0.84327106, places=6)

            self.assertAlmostEqual(results[2].beta, -0.5499320, places=6)
            self.assertAlmostEqual(results[2].standard_error, 0.3401110, places=6)
            self.assertAlmostEqual(results[2].t_stat, -1.616919, places=6)
            self.assertAlmostEqual(results[2].p_value, 0.24728705, places=6)

            self.assertAlmostEqual(results[3].beta, 1.09536219, places=6)
            self.assertAlmostEqual(results[3].standard_error, 0.6901002, places=6)
            self.assertAlmostEqual(results[3].t_stat, 1.5872510, places=6)
            self.assertAlmostEqual(results[3].p_value, 0.2533675, places=6)

    def test_linear_regression_with_dosage(self):
        covariates = hl.import_table(
            resource('regressionLinear.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        pheno = hl.import_table(
            resource('regressionLinear.pheno'), key='Sample', missing='0', types={'Pheno': hl.tfloat}
        )
        mt = hl.import_gen(resource('regressionLinear.gen'), sample_file=resource('regressionLinear.sample'))

        for linreg_function in self.linreg_functions:
            ht = linreg_function(
                y=pheno[mt.s].Pheno, x=hl.gp_dosage(mt.GP), covariates=[1.0, *list(covariates[mt.s].values())]
            )

            results = dict(hl.tuple([ht.locus.position, ht.row]).collect())

            self.assertAlmostEqual(results[1].beta, -0.29166985, places=4)
            self.assertAlmostEqual(results[1].standard_error, 1.2996510, places=4)
            self.assertAlmostEqual(results[1].t_stat, -0.22442167, places=6)
            self.assertAlmostEqual(results[1].p_value, 0.84327106, places=6)

            self.assertAlmostEqual(results[2].beta, -0.5499320, places=4)
            self.assertAlmostEqual(results[2].standard_error, 0.3401110, places=4)
            self.assertAlmostEqual(results[2].t_stat, -1.616919, places=6)
            self.assertAlmostEqual(results[2].p_value, 0.24728705, places=6)

            self.assertAlmostEqual(results[3].beta, 1.09536219, places=4)
            self.assertAlmostEqual(results[3].standard_error, 0.6901002, places=4)
            self.assertAlmostEqual(results[3].t_stat, 1.5872510, places=6)
            self.assertAlmostEqual(results[3].p_value, 0.2533675, places=6)
            self.assertTrue(np.isnan(results[6].standard_error))

    def test_linear_regression_equivalence_between_ds_and_gt(self):
        """Test that linear regressions on data converted from dosage to genotype returns the same results"""
        ds_mt = hl.import_vcf(resource('small-ds.vcf'))
        gt_mt = hl.import_vcf(resource('small-gt.vcf'))
        pheno_t = hl.read_table(resource('small-pheno.t'))
        ds_mt = ds_mt.annotate_cols(**pheno_t[ds_mt.s])
        gt_mt = gt_mt.annotate_cols(**pheno_t[gt_mt.s])

        for linreg_function in self.linreg_functions:
            ds_results_mt = linreg_function(y=ds_mt.phenotype, x=ds_mt.DS, covariates=[1.0])
            gt_results_mt = linreg_function(y=gt_mt.phenotype, x=gt_mt.GT.n_alt_alleles(), covariates=[1.0])
            ds_results_t = ds_results_mt.select(ds_p_value=ds_results_mt.p_value)
            gt_results_t = gt_results_mt.select(gt_p_value=gt_results_mt.p_value)
            results_t = ds_results_t.annotate(**gt_results_t[ds_results_t.locus, ds_results_t.alleles])
            self.assertTrue(all(hl.approx_equal(results_t.ds_p_value, results_t.gt_p_value, nan_same=True).collect()))

    def test_linear_regression_with_import_fam_boolean(self):
        covariates = hl.import_table(
            resource('regressionLinear.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        fam = hl.import_fam(resource('regressionLinear.fam'))
        mt = hl.import_vcf(resource('regressionLinear.vcf'))

        for linreg_function in self.linreg_functions:
            ht = linreg_function(
                y=fam[mt.s].is_case, x=mt.GT.n_alt_alleles(), covariates=[1.0, *list(covariates[mt.s].values())]
            )

            results = dict(hl.tuple([ht.locus.position, ht.row]).collect())

            self.assertAlmostEqual(results[1].beta, -0.28589421, places=6)
            self.assertAlmostEqual(results[1].standard_error, 1.2739153, places=6)
            self.assertAlmostEqual(results[1].t_stat, -0.22442167, places=6)
            self.assertAlmostEqual(results[1].p_value, 0.84327106, places=6)

            self.assertAlmostEqual(results[2].beta, -0.5417647, places=6)
            self.assertAlmostEqual(results[2].standard_error, 0.3350599, places=6)
            self.assertAlmostEqual(results[2].t_stat, -1.616919, places=6)
            self.assertAlmostEqual(results[2].p_value, 0.24728705, places=6)

            self.assertTrue(np.isnan(results[6].standard_error))
            self.assertTrue(np.isnan(results[7].standard_error))
            self.assertTrue(np.isnan(results[8].standard_error))
            self.assertTrue(np.isnan(results[9].standard_error))
            self.assertTrue(np.isnan(results[10].standard_error))

    def test_linear_regression_with_import_fam_quant(self):
        covariates = hl.import_table(
            resource('regressionLinear.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        fam = hl.import_fam(resource('regressionLinear.fam'), quant_pheno=True, missing='0')
        mt = hl.import_vcf(resource('regressionLinear.vcf'))

        for linreg_function in self.linreg_functions:
            ht = linreg_function(
                y=fam[mt.s].quant_pheno, x=mt.GT.n_alt_alleles(), covariates=[1.0, *list(covariates[mt.s].values())]
            )

            results = dict(hl.tuple([ht.locus.position, ht.row]).collect())

            self.assertAlmostEqual(results[1].beta, -0.28589421, places=6)
            self.assertAlmostEqual(results[1].standard_error, 1.2739153, places=6)
            self.assertAlmostEqual(results[1].t_stat, -0.22442167, places=6)
            self.assertAlmostEqual(results[1].p_value, 0.84327106, places=6)

            self.assertAlmostEqual(results[2].beta, -0.5417647, places=6)
            self.assertAlmostEqual(results[2].standard_error, 0.3350599, places=6)
            self.assertAlmostEqual(results[2].t_stat, -1.616919, places=6)
            self.assertAlmostEqual(results[2].p_value, 0.24728705, places=6)

            self.assertTrue(np.isnan(results[6].standard_error))
            self.assertTrue(np.isnan(results[7].standard_error))
            self.assertTrue(np.isnan(results[8].standard_error))
            self.assertTrue(np.isnan(results[9].standard_error))
            self.assertTrue(np.isnan(results[10].standard_error))

    def test_linear_regression_multi_pheno_same(self):
        covariates = hl.import_table(
            resource('regressionLinear.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        pheno = hl.import_table(
            resource('regressionLinear.pheno'), key='Sample', missing='0', types={'Pheno': hl.tfloat}
        )

        mt = hl.import_vcf(resource('regressionLinear.vcf'))

        for linreg_function in self.linreg_functions:
            single = linreg_function(
                y=pheno[mt.s].Pheno, x=mt.GT.n_alt_alleles(), covariates=list(covariates[mt.s].values())
            )
            multi = linreg_function(
                y=[pheno[mt.s].Pheno, pheno[mt.s].Pheno],
                x=mt.GT.n_alt_alleles(),
                covariates=list(covariates[mt.s].values()),
            )

            def eq(x1, x2):
                return (hl.is_nan(x1) & hl.is_nan(x2)) | (hl.abs(x1 - x2) < 1e-4)

            combined = single.annotate(multi=multi[single.key])
            self.assertTrue(
                combined.aggregate(
                    hl.agg.all(
                        eq(combined.p_value, combined.multi.p_value[0])
                        & eq(combined.multi.p_value[0], combined.multi.p_value[1])
                    )
                )
            )

    def test_logistic_regression_rows_max_iter_zero(self):
        import hail as hl

        mt = hl.utils.range_matrix_table(1, 3)
        mt = mt.annotate_entries(x=hl.literal([1, 1, 10]))
        try:
            ht = hl.logistic_regression_rows(
                test='wald', y=hl.literal([0, 0, 1])[mt.col_idx], x=mt.x[mt.col_idx], covariates=[1], max_iterations=0
            )
            ht.globals.collect()  # null model is a global
        except Exception as exc:
            assert (
                'Failed to fit logistic regression null model (standard MLE with covariates only): Newton iteration failed to converge'
                in exc.args[0]
            )
        else:
            assert False

    # Outside the spark backend, "logistic_regression_rows" automatically defers to the _ version.
    logreg_functions = (
        [hl.logistic_regression_rows, hl._logistic_regression_rows_nd]
        if backend_name == "spark"
        else [hl.logistic_regression_rows]
    )

    def test_logistic_regression_rows_max_iter_explodes(self):
        for logreg in self.logreg_functions:
            import hail as hl

            mt = hl.utils.range_matrix_table(1, 3)
            mt = mt.annotate_entries(x=hl.literal([1, 1, 10]))
            ht = logreg(
                test='wald', y=hl.literal([0, 0, 1])[mt.col_idx], x=mt.x[mt.col_idx], covariates=[1], max_iterations=100
            )
            fit = ht.collect()[0].fit
            assert fit.n_iterations < 100
            assert fit.exploded
            assert not fit.converged

    def test_firth_logistic_regression_rows_explodes_in_12_steps(self):
        import hail as hl

        mt = hl.utils.range_matrix_table(1, 3)
        mt = mt.annotate_entries(x=hl.literal([1, 1, 10]))
        ht = hl.logistic_regression_rows(
            test='firth', y=hl.literal([0, 1, 1, 0])[mt.col_idx], x=mt.x[mt.col_idx], covariates=[1], max_iterations=100
        )
        fit = ht.collect()[0].fit
        assert fit.n_iterations == 12
        assert fit.exploded
        assert not fit.converged

    def test_firth_logistic_regression_rows_does_not_converge_with_105_iterations(self):
        import hail as hl

        mt = hl.utils.range_matrix_table(1, 3)
        mt = mt.annotate_entries(x=hl.literal([1, 3, 10]))
        ht = hl.logistic_regression_rows(
            test='firth', y=hl.literal([0, 1, 1])[mt.col_idx], x=mt.x[mt.col_idx], covariates=[1], max_iterations=105
        )
        fit = ht.collect()[0].fit
        assert fit.n_iterations == 105
        assert not fit.exploded
        assert not fit.converged

    def test_firth_logistic_regression_rows_does_converge_with_more_iterations(self):
        import hail as hl

        mt = hl.utils.range_matrix_table(1, 3)
        mt = mt.annotate_entries(x=hl.literal([1, 3, 10]))
        ht = hl.logistic_regression_rows(
            test='firth',
            y=hl.literal([0, 1, 1])[mt.col_idx],
            x=mt.x[mt.col_idx],
            covariates=[1],
            max_iterations=106,
            tolerance=1e-6,
        )
        result = ht.collect()[0]
        fit = result.fit
        assert result.beta == pytest.approx(0.19699166375172233, abs=1e-14)
        assert result.chi_sq_stat == pytest.approx(0.6464918007192411, abs=1e-14)
        assert result.p_value == pytest.approx(0.4213697518249182, abs=1e-14)
        assert fit.n_iterations == 106
        assert not fit.exploded
        assert fit.converged

    def equal_with_nans(self, arr1, arr2):
        def both_nan_or_none(a, b):
            return (a is None or not np.isfinite(a)) and (b is None or not np.isfinite(b))

        return all([both_nan_or_none(a, b) or math.isclose(a, b) for a, b in zip(arr1, arr2)])

    @test_timeout(3 * 60)
    def test_weighted_linear_regression(self):
        covariates = hl.import_table(
            resource('regressionLinear.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        pheno = hl.import_table(
            resource('regressionLinear.pheno'), key='Sample', missing='0', types={'Pheno': hl.tfloat}
        )

        hl.import_table(
            resource('regressionLinear.weights'),
            key='Sample',
            missing='0',
            types={'Sample': hl.tstr, 'Weight1': hl.tfloat, 'Weight2': hl.tfloat},
        )

        mt = hl.import_vcf(resource('regressionLinear.vcf'))
        mt = mt.add_col_index()

        mt = mt.annotate_cols(y=hl.coalesce(pheno[mt.s].Pheno, 1.0))
        mt = mt.annotate_entries(x=hl.coalesce(mt.GT.n_alt_alleles(), 1.0))
        my_covs = [1.0, *list(covariates[mt.s].values())]

        ht_with_weights = hl._linear_regression_rows_nd(y=mt.y, x=mt.x, covariates=my_covs, weights=mt.col_idx)

        ht_pre_weighted_1 = hl._linear_regression_rows_nd(
            y=mt.y * hl.sqrt(mt.col_idx),
            x=mt.x * hl.sqrt(mt.col_idx),
            covariates=list(map(lambda e: e * hl.sqrt(mt.col_idx), my_covs)),
        )

        ht_pre_weighted_2 = hl._linear_regression_rows_nd(
            y=mt.y * hl.sqrt(mt.col_idx + 5),
            x=mt.x * hl.sqrt(mt.col_idx + 5),
            covariates=list(map(lambda e: e * hl.sqrt(mt.col_idx + 5), my_covs)),
        )

        ht_from_agg = mt.annotate_rows(
            my_linreg=hl.agg.linreg(mt.y, [1, mt.x, *list(covariates[mt.s].values())], weight=mt.col_idx)
        ).rows()

        betas_with_weights = ht_with_weights.beta.collect()
        betas_pre_weighted_1 = ht_pre_weighted_1.beta.collect()
        betas_pre_weighted_2 = ht_pre_weighted_2.beta.collect()

        betas_from_agg = ht_from_agg.my_linreg.beta[1].collect()

        assert self.equal_with_nans(betas_with_weights, betas_pre_weighted_1)
        assert self.equal_with_nans(betas_with_weights, betas_from_agg)

        ht_with_multiple_weights = hl._linear_regression_rows_nd(
            y=[[mt.y], [hl.abs(mt.y)]], x=mt.x, covariates=my_covs, weights=[mt.col_idx, mt.col_idx + 5]
        )

        # Check that preweighted 1 and preweighted 2 match up with fields 1 and 2 of multiple
        multi_weight_betas = ht_with_multiple_weights.beta.collect()
        multi_weight_betas_1 = [e[0][0] for e in multi_weight_betas]
        multi_weight_betas_2 = [e[1][0] for e in multi_weight_betas]

        assert np.array(multi_weight_betas).shape == (10, 2, 1)

        assert self.equal_with_nans(multi_weight_betas_1, betas_pre_weighted_1)
        assert self.equal_with_nans(multi_weight_betas_2, betas_pre_weighted_2)

    @test_timeout(3 * 60)
    def test_weighted_linear_regression__missing_weights_are_excluded(self):
        mt = hl.import_vcf(resource('regressionLinear.vcf'))
        pheno = hl.import_table(
            resource('regressionLinear.pheno'), key='Sample', missing='0', types={'Pheno': hl.tfloat}
        )
        mt = mt.annotate_cols(y=hl.coalesce(pheno[mt.s].Pheno, 1.0))
        weights = hl.import_table(
            resource('regressionLinear.weights'),
            key='Sample',
            missing='0',
            types={'Sample': hl.tstr, 'Weight1': hl.tfloat, 'Weight2': hl.tfloat},
        )
        mt = mt.annotate_entries(x=hl.coalesce(mt.GT.n_alt_alleles(), 1.0))
        ht_with_missing_weights = hl._linear_regression_rows_nd(
            y=[[mt.y], [hl.abs(mt.y)]], x=mt.x, covariates=[1], weights=[weights[mt.s].Weight1, weights[mt.s].Weight2]
        )

        mt_with_missing_weights = mt.annotate_cols(Weight1=weights[mt.s].Weight1, Weight2=weights[mt.s].Weight2)
        mt_with_missing_weight1_filtered = mt_with_missing_weights.filter_cols(
            hl.is_defined(mt_with_missing_weights.Weight1)
        )
        mt_with_missing_weight2_filtered = mt_with_missing_weights.filter_cols(
            hl.is_defined(mt_with_missing_weights.Weight2)
        )
        ht_from_agg_weight_1 = mt_with_missing_weight1_filtered.annotate_rows(
            my_linreg=hl.agg.linreg(
                mt_with_missing_weight1_filtered.y,
                [1, mt_with_missing_weight1_filtered.x],
                weight=weights[mt_with_missing_weight1_filtered.s].Weight1,
            )
        ).rows()
        ht_from_agg_weight_2 = mt_with_missing_weight2_filtered.annotate_rows(
            my_linreg=hl.agg.linreg(
                mt_with_missing_weight2_filtered.y,
                [1, mt_with_missing_weight2_filtered.x],
                weight=weights[mt_with_missing_weight2_filtered.s].Weight2,
            )
        ).rows()

        multi_weight_missing_results = ht_with_missing_weights.collect()
        multi_weight_missing_betas = [e.beta for e in multi_weight_missing_results]
        multi_weight_missing_betas_1 = [e[0][0] for e in multi_weight_missing_betas]
        multi_weight_missing_betas_2 = [e[1][0] for e in multi_weight_missing_betas]

        betas_from_agg_weight_1 = ht_from_agg_weight_1.my_linreg.beta[1].collect()
        betas_from_agg_weight_2 = ht_from_agg_weight_2.my_linreg.beta[1].collect()

        assert self.equal_with_nans(multi_weight_missing_betas_1, betas_from_agg_weight_1)
        assert self.equal_with_nans(multi_weight_missing_betas_2, betas_from_agg_weight_2)

        multi_weight_missing_p_values = [e.p_value for e in multi_weight_missing_results]
        multi_weight_missing_p_values_1 = [e[0][0] for e in multi_weight_missing_p_values]
        multi_weight_missing_p_values_2 = [e[1][0] for e in multi_weight_missing_p_values]

        p_values_from_agg_weight_1 = ht_from_agg_weight_1.my_linreg.p_value[1].collect()
        p_values_from_agg_weight_2 = ht_from_agg_weight_2.my_linreg.p_value[1].collect()

        assert self.equal_with_nans(multi_weight_missing_p_values_1, p_values_from_agg_weight_1)
        assert self.equal_with_nans(multi_weight_missing_p_values_2, p_values_from_agg_weight_2)

        multi_weight_missing_t_stats = [e.t_stat for e in multi_weight_missing_results]
        multi_weight_missing_t_stats_1 = [e[0][0] for e in multi_weight_missing_t_stats]
        multi_weight_missing_t_stats_2 = [e[1][0] for e in multi_weight_missing_t_stats]

        t_stats_from_agg_weight_1 = ht_from_agg_weight_1.my_linreg.t_stat[1].collect()
        t_stats_from_agg_weight_2 = ht_from_agg_weight_2.my_linreg.t_stat[1].collect()

        assert self.equal_with_nans(multi_weight_missing_t_stats_1, t_stats_from_agg_weight_1)
        assert self.equal_with_nans(multi_weight_missing_t_stats_2, t_stats_from_agg_weight_2)

        multi_weight_missing_se = [e.standard_error for e in multi_weight_missing_results]
        multi_weight_missing_se_1 = [e[0][0] for e in multi_weight_missing_se]
        multi_weight_missing_se_2 = [e[1][0] for e in multi_weight_missing_se]

        se_from_agg_weight_1 = ht_from_agg_weight_1.my_linreg.standard_error[1].collect()
        se_from_agg_weight_2 = ht_from_agg_weight_2.my_linreg.standard_error[1].collect()

        assert self.equal_with_nans(multi_weight_missing_se_1, se_from_agg_weight_1)
        assert self.equal_with_nans(multi_weight_missing_se_2, se_from_agg_weight_2)

    @test_timeout(3 * 60)
    def test_errors_weighted_linear_regression(self):
        mt = hl.utils.range_matrix_table(20, 10).annotate_entries(x=2)
        mt = mt.annotate_cols(**{f"col_{i}": i for i in range(4)})

        self.assertRaises(
            ValueError,
            lambda: hl._linear_regression_rows_nd(y=[[mt.col_1]], x=mt.x, covariates=[1], weights=[mt.col_2, mt.col_3]),
        )

        self.assertRaises(
            ValueError, lambda: hl._linear_regression_rows_nd(y=[mt.col_1], x=mt.x, covariates=[1], weights=[mt.col_2])
        )

        self.assertRaises(
            ValueError, lambda: hl._linear_regression_rows_nd(y=[[mt.col_1]], x=mt.x, covariates=[1], weights=mt.col_2)
        )

    # comparing to R:
    # x = c(0, 1, 0, 0, 0, 1, 0, 0, 0, 0)
    # y = c(0, 0, 1, 1, 1, 1, 0, 0, 1, 1)
    # c1 = c(0, 2, 1, -2, -2, 4, 1, 2, 3, 4)
    # c2 = c(-1, 3, 5, 0, -4, 3, 0, -2, -1, -4)
    # logfit <- glm(y ~ x + c1 + c2, family=binomial(link="logit"))
    # waldtest <- coef(summary(logfit))
    # beta <- waldtest["x", "Estimate"]
    # se <- waldtest["x", "Std. Error"]
    # zstat <- waldtest["x", "z value"]
    # pval <- waldtest["x", "Pr(>|z|)"]
    def test_logistic_regression_wald_test(self):
        covariates = hl.import_table(
            resource('regressionLogistic.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        pheno = hl.import_table(
            resource('regressionLogisticBoolean.pheno'), key='Sample', missing='0', types={'isCase': hl.tbool}
        )
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))

        for logistic_regression_function in self.logreg_functions:
            ht = logistic_regression_function(
                'wald',
                y=pheno[mt.s].isCase,
                x=mt.GT.n_alt_alleles(),
                covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2],
            )

            results = dict(hl.tuple([ht.locus.position, ht.row]).collect())

            self.assertAlmostEqual(results[1].beta, -0.81226793796, places=6)
            self.assertAlmostEqual(results[1].standard_error, 2.1085483421, places=6)
            self.assertAlmostEqual(results[1].z_stat, -0.3852261396, places=6)
            self.assertAlmostEqual(results[1].p_value, 0.7000698784, places=6)

            self.assertAlmostEqual(results[2].beta, -0.43659460858, places=6)
            self.assertAlmostEqual(results[2].standard_error, 1.0296902941, places=6)
            self.assertAlmostEqual(results[2].z_stat, -0.4240057531, places=6)
            self.assertAlmostEqual(results[2].p_value, 0.6715616176, places=6)

            def is_constant(r):
                return (not r.fit.converged) or np.isnan(r.p_value) or abs(r.p_value - 1) < 1e-4

            self.assertFalse(results[3].fit.converged)  # separable
            self.assertTrue(is_constant(results[6]))
            self.assertTrue(is_constant(results[7]))
            self.assertTrue(is_constant(results[8]))
            self.assertTrue(is_constant(results[9]))
            self.assertTrue(is_constant(results[10]))

    def test_logistic_regression_wald_test_apply_multi_pheno(self):
        covariates = hl.import_table(
            resource('regressionLogistic.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        pheno = hl.import_table(
            resource('regressionLogisticBoolean.pheno'), key='Sample', missing='0', types={'isCase': hl.tbool}
        )
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))

        for logistic_regression_function in self.logreg_functions:
            ht = logistic_regression_function(
                'wald',
                y=[pheno[mt.s].isCase],
                x=mt.GT.n_alt_alleles(),
                covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2],
            )

            results = dict(hl.tuple([ht.locus.position, ht.row]).collect())

            self.assertEqual(len(results[1].logistic_regression), 1)
            self.assertAlmostEqual(results[1].logistic_regression[0].beta, -0.81226793796, places=6)
            self.assertAlmostEqual(results[1].logistic_regression[0].standard_error, 2.1085483421, places=6)
            self.assertAlmostEqual(results[1].logistic_regression[0].z_stat, -0.3852261396, places=6)
            self.assertAlmostEqual(results[1].logistic_regression[0].p_value, 0.7000698784, places=6)

            self.assertEqual(len(results[2].logistic_regression), 1)
            self.assertAlmostEqual(results[2].logistic_regression[0].beta, -0.43659460858, places=6)
            self.assertAlmostEqual(results[2].logistic_regression[0].standard_error, 1.0296902941, places=6)
            self.assertAlmostEqual(results[2].logistic_regression[0].z_stat, -0.4240057531, places=6)
            self.assertAlmostEqual(results[2].logistic_regression[0].p_value, 0.6715616176, places=6)

            def is_constant(r):
                return (
                    (not r.logistic_regression[0].fit.converged)
                    or np.isnan(r.logistic_regression[0].p_value)
                    or abs(r.logistic_regression[0].p_value - 1) < 1e-4
                )

            self.assertEqual(len(results[3].logistic_regression), 1)
            self.assertFalse(results[3].logistic_regression[0].fit.converged)  # separable
            self.assertTrue(is_constant(results[6]))
            self.assertTrue(is_constant(results[7]))
            self.assertTrue(is_constant(results[8]))
            self.assertTrue(is_constant(results[9]))
            self.assertTrue(is_constant(results[10]))

    def test_logistic_regression_wald_test_multi_pheno_bgen_dosage(self):
        covariates = hl.import_table(
            resource('regressionLogisticMultiPheno.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        ).cache()
        pheno = hl.import_table(
            resource('regressionLogisticMultiPheno.pheno'),
            key='Sample',
            missing='NA',
            types={'Pheno1': hl.tint32, 'Pheno2': hl.tint32},
        ).cache()
        bgen_path = new_temp_file(extension='bgen')
        Env.fs().copy(resource('example.8bits.bgen'), bgen_path)

        hl.index_bgen(bgen_path, contig_recoding={'01': '1'}, reference_genome='GRCh37')

        mt = hl.import_bgen(bgen_path, entry_fields=['dosage'])

        for logistic_regression_function in self.logreg_functions:
            ht_single_pheno = logistic_regression_function(
                'wald',
                y=pheno[mt.s].Pheno1,
                x=mt.dosage,
                covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2],
            )

            ht_multi_pheno = logistic_regression_function(
                'wald',
                y=[pheno[mt.s].Pheno1, pheno[mt.s].Pheno2],
                x=mt.dosage,
                covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2],
            )

            single_results = dict(hl.tuple([ht_single_pheno.locus.position, ht_single_pheno.row]).collect())
            multi_results = dict(hl.tuple([ht_multi_pheno.locus.position, ht_multi_pheno.row]).collect())
            self.assertEqual(len(multi_results[1001].logistic_regression), 2)
            self.assertAlmostEqual(multi_results[1001].logistic_regression[0].beta, single_results[1001].beta, places=6)
            self.assertAlmostEqual(
                multi_results[1001].logistic_regression[0].standard_error, single_results[1001].standard_error, places=6
            )
            self.assertAlmostEqual(
                multi_results[1001].logistic_regression[0].z_stat, single_results[1001].z_stat, places=6
            )
            self.assertAlmostEqual(
                multi_results[1001].logistic_regression[0].p_value, single_results[1001].p_value, places=6
            )
            # TODO test handling of missingness

    def test_logistic_regression_wald_test_pl(self):
        covariates = hl.import_table(
            resource('regressionLogistic.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        pheno = hl.import_table(
            resource('regressionLogisticBoolean.pheno'), key='Sample', missing='0', types={'isCase': hl.tbool}
        )
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))

        for logistic_regression_function in self.logreg_functions:
            ht = logistic_regression_function(
                test='wald',
                y=pheno[mt.s].isCase,
                x=hl.pl_dosage(mt.PL),
                covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2],
            )

            results = dict(hl.tuple([ht.locus.position, ht.row]).collect())

            self.assertAlmostEqual(results[1].beta, -0.8286774, places=6)
            self.assertAlmostEqual(results[1].standard_error, 2.151145, places=6)
            self.assertAlmostEqual(results[1].z_stat, -0.3852261, places=6)
            self.assertAlmostEqual(results[1].p_value, 0.7000699, places=6)

            self.assertAlmostEqual(results[2].beta, -0.4431764, places=6)
            self.assertAlmostEqual(results[2].standard_error, 1.045213, places=6)
            self.assertAlmostEqual(results[2].z_stat, -0.4240058, places=6)
            self.assertAlmostEqual(results[2].p_value, 0.6715616, places=6)

            def is_constant(r):
                return (not r.fit.converged) or np.isnan(r.p_value) or abs(r.p_value - 1) < 1e-4

            self.assertFalse(results[3].fit.converged)  # separable
            self.assertTrue(is_constant(results[6]))
            self.assertTrue(is_constant(results[7]))
            self.assertTrue(is_constant(results[8]))
            self.assertTrue(is_constant(results[9]))
            self.assertTrue(is_constant(results[10]))

    def test_logistic_regression_wald_dosage(self):
        covariates = hl.import_table(
            resource('regressionLogistic.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        pheno = hl.import_table(
            resource('regressionLogisticBoolean.pheno'), key='Sample', missing='0', types={'isCase': hl.tbool}
        )
        mt = hl.import_gen(resource('regressionLogistic.gen'), sample_file=resource('regressionLogistic.sample'))

        for logistic_regression_function in self.logreg_functions:
            ht = logistic_regression_function(
                test='wald',
                y=pheno[mt.s].isCase,
                x=hl.gp_dosage(mt.GP),
                covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2],
            )

            results = dict(hl.tuple([ht.locus.position, ht.row]).collect())

            self.assertAlmostEqual(results[1].beta, -0.8286774, places=4)
            self.assertAlmostEqual(results[1].standard_error, 2.151145, places=4)
            self.assertAlmostEqual(results[1].z_stat, -0.3852261, places=4)
            self.assertAlmostEqual(results[1].p_value, 0.7000699, places=4)

            self.assertAlmostEqual(results[2].beta, -0.4431764, places=4)
            self.assertAlmostEqual(results[2].standard_error, 1.045213, places=4)
            self.assertAlmostEqual(results[2].z_stat, -0.4240058, places=4)
            self.assertAlmostEqual(results[2].p_value, 0.6715616, places=4)

            def is_constant(r):
                return (not r.fit.converged) or np.isnan(r.p_value) or abs(r.p_value - 1) < 1e-4

            self.assertFalse(results[3].fit.converged)  # separable
            self.assertTrue(is_constant(results[6]))
            self.assertTrue(is_constant(results[7]))
            self.assertTrue(is_constant(results[8]))
            self.assertTrue(is_constant(results[9]))
            self.assertTrue(is_constant(results[10]))

    # comparing to output of R code:
    # x = c(0, 1, 0, 0, 0, 1, 0, 0, 0, 0)
    # y = c(0, 0, 1, 1, 1, 1, 0, 0, 1, 1)
    # c1 = c(0, 2, 1, -2, -2, 4, 1, 2, 3, 4)
    # c2 = c(-1, 3, 5, 0, -4, 3, 0, -2, -1, -4)
    # logfit <- glm(y ~ x + c1 + c2, family=binomial(link="logit"))
    # logfitnull <- glm(y ~ c1 + c2, family=binomial(link="logit"))
    # beta <- coef(summary(logfit))["x", "Estimate"]
    # lrtest <- anova(logfitnull, logfit, test="LRT")
    # chi2 <- lrtest[["Deviance"]][2]
    # pval <- lrtest[["Pr(>Chi)"]][2]
    def test_logistic_regression_lrt(self):
        covariates = hl.import_table(
            resource('regressionLogistic.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        pheno = hl.import_table(
            resource('regressionLogisticBoolean.pheno'), key='Sample', missing='0', types={'isCase': hl.tbool}
        )
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))

        for logistic_regression_function in self.logreg_functions:
            ht = logistic_regression_function(
                test='lrt',
                y=pheno[mt.s].isCase,
                x=mt.GT.n_alt_alleles(),
                covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2],
            )

            results = dict(hl.tuple([ht.locus.position, ht.row]).collect())

            self.assertAlmostEqual(results[1].beta, -0.81226793796, places=6)
            self.assertAlmostEqual(results[1].chi_sq_stat, 0.1503349167, places=6)
            self.assertAlmostEqual(results[1].p_value, 0.6982155052, places=6)

            self.assertAlmostEqual(results[2].beta, -0.43659460858, places=6)
            self.assertAlmostEqual(results[2].chi_sq_stat, 0.1813968574, places=6)
            self.assertAlmostEqual(results[2].p_value, 0.6701755415, places=6)

            def is_constant(r):
                return (not r.fit.converged) or np.isnan(r.p_value) or abs(r.p_value - 1) < 1e-4

            self.assertFalse(results[3].fit.converged)  # separable
            self.assertTrue(is_constant(results[6]))
            self.assertTrue(is_constant(results[7]))
            self.assertTrue(is_constant(results[8]))
            self.assertTrue(is_constant(results[9]))
            self.assertTrue(is_constant(results[10]))

    # comparing to output of R code:
    # x = c(0, 1, 0, 0, 0, 1, 0, 0, 0, 0)
    # y = c(0, 0, 1, 1, 1, 1, 0, 0, 1, 1)
    # c1 = c(0, 2, 1, -2, -2, 4, 1, 2, 3, 4)
    # c2 = c(-1, 3, 5, 0, -4, 3, 0, -2, -1, -4)
    # logfit <- glm(y ~ c1 + c2 + x, family=binomial(link="logit"))
    # logfitnull <- glm(y ~ c1 + c2, family=binomial(link="logit"))
    # scoretest <- anova(logfitnull, logfit, test="Rao")
    # chi2 <- scoretest[["Rao"]][2]
    # pval <- scoretest[["Pr(>Chi)"]][2]
    def test_logistic_regression_score(self):
        covariates = hl.import_table(
            resource('regressionLogistic.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        pheno = hl.import_table(
            resource('regressionLogisticBoolean.pheno'), key='Sample', missing='0', types={'isCase': hl.tbool}
        )
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))

        def is_constant(r):
            return r.chi_sq_stat is None or r.chi_sq_stat < 1e-6

        for logreg_function in self.logreg_functions:
            ht = logreg_function(
                test='score',
                y=pheno[mt.s].isCase,
                x=mt.GT.n_alt_alleles(),
                covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2],
            )

            results = dict(hl.tuple([ht.locus.position, ht.row]).collect())

            self.assertAlmostEqual(results[1].chi_sq_stat, 0.1502364955, places=6)
            self.assertAlmostEqual(results[1].p_value, 0.6983094571, places=6)

            self.assertAlmostEqual(results[2].chi_sq_stat, 0.1823600965, places=6)
            self.assertAlmostEqual(results[2].p_value, 0.6693528073, places=6)

            self.assertAlmostEqual(results[3].chi_sq_stat, 7.047367694, places=6)
            self.assertAlmostEqual(results[3].p_value, 0.007938182229, places=6)

            self.assertTrue(is_constant(results[6]))
            self.assertTrue(is_constant(results[7]))
            self.assertTrue(is_constant(results[8]))
            self.assertTrue(is_constant(results[9]))
            self.assertTrue(is_constant(results[10]))

    def test_logreg_pass_through(self):
        covariates = hl.import_table(
            resource('regressionLogistic.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        pheno = hl.import_table(
            resource('regressionLogisticBoolean.pheno'), key='Sample', missing='0', types={'isCase': hl.tbool}
        )
        mt = hl.import_vcf(resource('regressionLogistic.vcf')).annotate_rows(foo=hl.struct(bar=hl.rand_norm(0, 1)))

        for logreg_function in self.logreg_functions:
            ht = logreg_function(
                'wald',
                y=pheno[mt.s].isCase,
                x=mt.GT.n_alt_alleles(),
                covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2],
                pass_through=['filters', mt.foo.bar, mt.qual],
            )

        assert mt.aggregate_rows(hl.agg.all(mt.foo.bar == ht[mt.row_key].bar))

    # comparing to R:
    # x = c(0, 1, 0, 0, 0, 1, 0, 0, 0, 0)
    # y = c(0, 2, 5, 3, 6, 2, 1, 1, 0, 0)
    # c1 = c(0, 2, 1, -2, -2, 4, 1, 2, 3, 4)
    # c2 = c(-1, 3, 5, 0, -4, 3, 0, -2, -1, -4)
    # logfit <- glm(y ~ x + c1 + c2, family=poisson(link="log"))
    # waldtest <- coef(summary(logfit))
    # beta <- waldtest["x", "Estimate"]
    # se <- waldtest["x", "Std. Error"]
    # zstat <- waldtest["x", "z value"]
    # pval <- waldtest["x", "Pr(>|z|)"]
    def test_poission_regression_wald_test(self):
        covariates = hl.import_table(
            resource('regressionLogistic.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        pheno = hl.import_table(
            resource('regressionPoisson.pheno'), key='Sample', missing='-1', types={'count': hl.tint32}
        )
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))
        ht = hl.poisson_regression_rows(
            test='wald',
            y=pheno[mt.s].count,
            x=mt.GT.n_alt_alleles(),
            covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2],
        )

        results = dict(hl.tuple([ht.locus.position, ht.row]).collect())

        self.assertAlmostEqual(results[1].beta, 0.6725210143, places=6)
        self.assertAlmostEqual(results[1].standard_error, 0.7265562271, places=5)
        self.assertAlmostEqual(results[1].z_stat, 0.9256283123, places=5)
        self.assertAlmostEqual(results[1].p_value, 0.3546391746, places=6)

        self.assertAlmostEqual(results[2].beta, -0.5025904503, places=6)
        self.assertAlmostEqual(results[2].standard_error, 0.3549856127, places=5)
        self.assertAlmostEqual(results[2].z_stat, -1.415805126, places=5)
        self.assertAlmostEqual(results[2].p_value, 0.1568325682, places=6)

        def is_constant(r):
            return (not r.fit.converged) or np.isnan(r.p_value) or abs(r.p_value - 1) < 1e-4

        self.assertTrue(is_constant(results[6]))
        self.assertTrue(is_constant(results[7]))
        self.assertTrue(is_constant(results[8]))
        self.assertTrue(is_constant(results[9]))
        self.assertTrue(is_constant(results[10]))

    def test_poisson_regression_max_iterations(self):
        import hail as hl

        mt = hl.utils.range_matrix_table(1, 3)
        mt = mt.annotate_entries(x=hl.literal([1, 3, 10, 5]))
        ht = hl.poisson_regression_rows(
            'wald', y=hl.literal([0, 1, 1, 0])[mt.col_idx], x=mt.x[mt.col_idx], covariates=[1], max_iterations=1
        )
        fit = ht.collect()[0].fit
        assert fit.n_iterations == 1
        assert not fit.converged
        assert not fit.exploded

    # comparing to R:
    # x = c(0, 1, 0, 0, 0, 1, 0, 0, 0, 0)
    # y = c(0, 2, 5, 3, 6, 2, 1, 1, 0, 0)
    # c1 = c(0, 2, 1, -2, -2, 4, 1, 2, 3, 4)
    # c2 = c(-1, 3, 5, 0, -4, 3, 0, -2, -1, -4)
    # poisfit <- glm(y ~ x + c1 + c2, family=poisson(link="log"))
    # poisfitnull <- glm(y ~ c1 + c2, family=poisson(link="log"))
    # beta <- coef(summary(poisfit))["x", "Estimate"]
    # lrtest <- anova(poisfitnull, poisfit, test="LRT")
    # chi2 <- lrtest[["Deviance"]][2]
    # pval <- lrtest[["Pr(>Chi)"]][2]
    def test_poisson_regression_lrt(self):
        covariates = hl.import_table(
            resource('regressionLogistic.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        pheno = hl.import_table(
            resource('regressionPoisson.pheno'), key='Sample', missing='-1', types={'count': hl.tint32}
        )
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))
        ht = hl.poisson_regression_rows(
            test='lrt',
            y=pheno[mt.s].count,
            x=mt.GT.n_alt_alleles(),
            covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2],
        )

        results = dict(hl.tuple([ht.locus.position, ht.row]).collect())

        self.assertAlmostEqual(results[1].beta, 0.6725210143, places=6)
        self.assertAlmostEqual(results[1].chi_sq_stat, 0.8334198333, places=5)
        self.assertAlmostEqual(results[1].p_value, 0.361285509, places=6)

        self.assertAlmostEqual(results[2].beta, -0.5025904503, places=6)
        self.assertAlmostEqual(results[2].chi_sq_stat, 2.193682097, places=5)
        self.assertAlmostEqual(results[2].p_value, 0.1385776894, places=6)

        def is_constant(r):
            return (not r.fit.converged) or np.isnan(r.p_value) or abs(r.p_value - 1) < 1e-4

        self.assertTrue(is_constant(results[6]))
        self.assertTrue(is_constant(results[7]))
        self.assertTrue(is_constant(results[8]))
        self.assertTrue(is_constant(results[9]))
        self.assertTrue(is_constant(results[10]))

    # comparing to R:
    # x = c(0, 1, 0, 0, 0, 1, 0, 0, 0, 0)
    # y = c(0, 2, 5, 3, 6, 2, 1, 1, 0, 0)
    # c1 = c(0, 2, 1, -2, -2, 4, 1, 2, 3, 4)
    # c2 = c(-1, 3, 5, 0, -4, 3, 0, -2, -1, -4)
    # poisfit <- glm(y ~ c1 + c2 + x, family=poisson(link="log"))
    # poisfitnull <- glm(y ~ c1 + c2, family=poisson(link="log"))
    # scoretest <- anova(poisfitnull, poisfit, test="Rao")
    # chi2 <- scoretest[["Rao"]][2]
    # pval <- scoretest[["Pr(>Chi)"]][2]
    def test_poisson_regression_score_test(self):
        covariates = hl.import_table(
            resource('regressionLogistic.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        pheno = hl.import_table(
            resource('regressionPoisson.pheno'), key='Sample', missing='-1', types={'count': hl.tint32}
        )
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))
        ht = hl.poisson_regression_rows(
            test='score',
            y=pheno[mt.s].count,
            x=mt.GT.n_alt_alleles(),
            covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2],
        )

        results = dict(hl.tuple([ht.locus.position, ht.row]).collect())

        self.assertAlmostEqual(results[1].chi_sq_stat, 0.8782455145, places=4)
        self.assertAlmostEqual(results[1].p_value, 0.3486826695, places=5)

        self.assertAlmostEqual(results[2].chi_sq_stat, 2.067574259, places=4)
        self.assertAlmostEqual(results[2].p_value, 0.1504606684, places=5)

        self.assertAlmostEqual(results[3].chi_sq_stat, 5.483930429, places=4)
        self.assertAlmostEqual(results[3].p_value, 0.01919205854, places=5)

        def is_constant(r):
            return r.chi_sq_stat is None or r.chi_sq_stat < 1e-6

        self.assertTrue(is_constant(results[6]))
        self.assertTrue(is_constant(results[7]))
        self.assertTrue(is_constant(results[8]))
        self.assertTrue(is_constant(results[9]))
        self.assertTrue(is_constant(results[10]))

    def test_poisson_pass_through(self):
        covariates = hl.import_table(
            resource('regressionLogistic.cov'), key='Sample', types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat}
        )
        pheno = hl.import_table(
            resource('regressionPoisson.pheno'), key='Sample', missing='-1', types={'count': hl.tint32}
        )
        mt = hl.import_vcf(resource('regressionLogistic.vcf')).annotate_rows(foo=hl.struct(bar=hl.rand_norm(0, 1)))
        ht = hl.poisson_regression_rows(
            test='wald',
            y=pheno[mt.s].count,
            x=mt.GT.n_alt_alleles(),
            covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2],
            pass_through=['filters', mt.foo.bar, mt.qual],
        )

        assert mt.aggregate_rows(hl.agg.all(mt.foo.bar == ht[mt.row_key].bar))

    def test_genetic_relatedness_matrix(self):
        n, m = 100, 200
        mt = hl.balding_nichols_model(3, n, m, fst=[0.9, 0.9, 0.9], n_partitions=4)

        g = BlockMatrix.from_entry_expr(mt.GT.n_alt_alleles()).to_numpy().T

        col_means = np.mean(g, axis=0, keepdims=True)
        col_filter = np.logical_and(col_means > 0, col_means < 2)

        g = g[:, np.squeeze(col_filter)]
        col_means = col_means[col_filter]
        col_sd_hwe = np.sqrt(col_means * (1 - col_means / 2))
        g_std = (g - col_means) / col_sd_hwe

        m1 = g_std.shape[1]
        self.assertTrue(m1 < m)
        k = (g_std @ g_std.T) / m1

        rrm = hl.genetic_relatedness_matrix(mt.GT).to_numpy()

        self.assertTrue(np.allclose(k, rrm))

    @staticmethod
    def _filter_and_standardize_cols(a):
        a = a.copy()
        col_means = np.mean(a, axis=0, keepdims=True)
        a -= col_means
        col_lengths = np.linalg.norm(a, axis=0, keepdims=True)
        col_filter = col_lengths > 0
        return np.copy(a[:, np.squeeze(col_filter)] / col_lengths[col_filter])

    def test_realized_relationship_matrix(self):
        n, m = 100, 200
        hl.reset_global_randomness()
        mt = hl.balding_nichols_model(3, n, m, fst=[0.9, 0.9, 0.9], n_partitions=4)

        g = BlockMatrix.from_entry_expr(mt.GT.n_alt_alleles()).to_numpy().T
        g_std = self._filter_and_standardize_cols(g)
        m1 = g_std.shape[1]
        self.assertTrue(m1 < m)
        k = (g_std @ g_std.T) * (n / m1)

        rrm = hl.realized_relationship_matrix(mt.GT).to_numpy()
        self.assertTrue(np.allclose(k, rrm))

        one_sample = hl.balding_nichols_model(1, 1, 10)
        self.assertRaises(FatalError, lambda: hl.realized_relationship_matrix(one_sample.GT))

    def test_row_correlation_vs_hardcode(self):
        data = [
            {'v': '1:1:A:C', 's': '1', 'GT': hl.Call([0, 0])},
            {'v': '1:1:A:C', 's': '2', 'GT': hl.Call([0, 0])},
            {'v': '1:1:A:C', 's': '3', 'GT': hl.Call([0, 1])},
            {'v': '1:1:A:C', 's': '4', 'GT': hl.Call([1, 1])},
            {'v': '1:2:G:T', 's': '1', 'GT': hl.Call([0, 1])},
            {'v': '1:2:G:T', 's': '2', 'GT': hl.Call([1, 1])},
            {'v': '1:2:G:T', 's': '3', 'GT': hl.Call([0, 1])},
            {'v': '1:2:G:T', 's': '4', 'GT': hl.Call([0, 0])},
            {'v': '1:3:C:G', 's': '1', 'GT': hl.Call([0, 1])},
            {'v': '1:3:C:G', 's': '2', 'GT': hl.Call([0, 0])},
            {'v': '1:3:C:G', 's': '3', 'GT': hl.Call([1, 1])},
            {'v': '1:3:C:G', 's': '4', 'GT': hl.missing(hl.tcall)},
        ]
        ht = hl.Table.parallelize(data, hl.dtype('struct{v: str, s: str, GT: call}'))
        mt = ht.to_matrix_table(['v'], ['s'])

        actual = hl.row_correlation(mt.GT.n_alt_alleles()).to_numpy()
        expected = [[1.0, -0.85280287, 0.42640143], [-0.85280287, 1.0, -0.5], [0.42640143, -0.5, 1.0]]

        self.assertTrue(np.allclose(actual, expected))

    def test_row_correlation_vs_numpy(self):
        n_samples, n_variants = 11, 10
        mt = hl.balding_nichols_model(3, n_samples, n_variants, n_partitions=2)
        mt = mt.annotate_rows(sd=agg.stats(mt.GT.n_alt_alleles()).stdev)
        mt = mt.filter_rows(mt.sd > 1e-30)

        g = BlockMatrix.from_entry_expr(mt.GT.n_alt_alleles()).to_numpy().T
        g_std = self._filter_and_standardize_cols(g)
        l = g_std.T @ g_std

        cor = hl.row_correlation(mt.GT.n_alt_alleles()).to_numpy()

        self.assertGreater(cor.shape[0], 5)
        self.assertEqual(cor.shape[0], cor.shape[1])
        self.assertTrue(np.allclose(l, cor))

    def get_ld_matrix_mt(self):
        data = [
            {'v': '1:1:A:C', 'cm': 0.1, 's': 'a', 'GT': hl.Call([0, 0])},
            {'v': '1:1:A:C', 'cm': 0.1, 's': 'b', 'GT': hl.Call([0, 0])},
            {'v': '1:1:A:C', 'cm': 0.1, 's': 'c', 'GT': hl.Call([0, 1])},
            {'v': '1:1:A:C', 'cm': 0.1, 's': 'd', 'GT': hl.Call([1, 1])},
            {'v': '1:2000000:G:T', 'cm': 0.9, 's': 'a', 'GT': hl.Call([0, 1])},
            {'v': '1:2000000:G:T', 'cm': 0.9, 's': 'b', 'GT': hl.Call([1, 1])},
            {'v': '1:2000000:G:T', 'cm': 0.9, 's': 'c', 'GT': hl.Call([0, 1])},
            {'v': '1:2000000:G:T', 'cm': 0.9, 's': 'd', 'GT': hl.Call([0, 0])},
            {'v': '2:1:C:G', 'cm': 0.2, 's': 'a', 'GT': hl.Call([0, 1])},
            {'v': '2:1:C:G', 'cm': 0.2, 's': 'b', 'GT': hl.Call([0, 0])},
            {'v': '2:1:C:G', 'cm': 0.2, 's': 'c', 'GT': hl.Call([1, 1])},
            {'v': '2:1:C:G', 'cm': 0.2, 's': 'd', 'GT': hl.missing(hl.tcall)},
        ]
        ht = hl.Table.parallelize(data, hl.dtype('struct{v: str, s: str, cm: float64, GT: call}'))
        ht = ht.transmute(**hl.parse_variant(ht.v))
        return ht.to_matrix_table(row_key=['locus', 'alleles'], col_key=['s'], row_fields=['cm'])

    def test_ld_matrix_1(self):
        mt = self.get_ld_matrix_mt()
        self.assertTrue(
            np.allclose(
                hl.ld_matrix(mt.GT.n_alt_alleles(), mt.locus, radius=1e6).to_numpy(),
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            )
        )

    def test_ld_matrix_2(self):
        mt = self.get_ld_matrix_mt()
        self.assertTrue(
            np.allclose(
                hl.ld_matrix(mt.GT.n_alt_alleles(), mt.locus, radius=2e6).to_numpy(),
                [[1.0, -0.85280287, 0.0], [-0.85280287, 1.0, 0.0], [0.0, 0.0, 1.0]],
            )
        )

    def test_ld_matrix_3(self):
        mt = self.get_ld_matrix_mt()
        self.assertTrue(
            np.allclose(
                hl.ld_matrix(mt.GT.n_alt_alleles(), mt.locus, radius=0.5, coord_expr=mt.cm).to_numpy(),
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            )
        )

    def test_ld_matrix_4(self):
        mt = self.get_ld_matrix_mt()
        self.assertTrue(
            np.allclose(
                hl.ld_matrix(mt.GT.n_alt_alleles(), mt.locus, radius=1.0, coord_expr=mt.cm).to_numpy(),
                [[1.0, -0.85280287, 0.0], [-0.85280287, 1.0, 0.0], [0.0, 0.0, 1.0]],
            )
        )

    @qobtest
    def test_split_multi_hts(self):
        ds1 = hl.import_vcf(resource('split_test.vcf'))
        ds1 = hl.split_multi_hts(ds1)
        ds2 = hl.import_vcf(resource('split_test_b.vcf'))
        df = ds1.rows()
        self.assertTrue(df.all((df.locus.position == 1180) | df.was_split))
        ds1 = ds1.drop('was_split', 'a_index')
        self.assertTrue(ds1._same(ds2))

    @qobtest
    def test_split_multi_table(self):
        ds1 = hl.import_vcf(resource('split_test.vcf')).rows()
        ds1 = hl.split_multi(ds1)
        ds2 = hl.import_vcf(resource('split_test_b.vcf')).rows()
        self.assertTrue(ds1.all((ds1.locus.position == 1180) | ds1.was_split))
        ds1 = ds1.drop('was_split', 'a_index', 'old_locus', 'old_alleles')
        self.assertTrue(ds1._same(ds2))

        ds1 = hl.import_vcf(resource('split_test.vcf')).rows()
        ds1 = hl.split_multi_hts(ds1)
        ds2 = hl.import_vcf(resource('split_test_b.vcf')).rows()
        self.assertTrue(ds1.all((ds1.locus.position == 1180) | ds1.was_split))
        ds1 = ds1.drop('was_split', 'a_index')
        self.assertTrue(ds1._same(ds2))

    @qobtest
    def test_split_multi_shuffle(self):
        ht = hl.utils.range_table(1)
        ht = ht.annotate(
            keys=[
                hl.struct(locus=hl.locus('1', 1180), alleles=['A', 'C', 'T']),
                hl.struct(locus=hl.locus('1', 1180), alleles=['A', 'G']),
            ]
        )
        ht = ht.explode(ht.keys)
        ht = ht.key_by(**ht.keys).drop('keys')
        alleles = hl.split_multi(ht, permit_shuffle=True).alleles.collect()
        assert alleles == [['A', 'C'], ['A', 'G'], ['A', 'T']]

        ht = ht.annotate_globals(cols=[hl.struct(s='sample1'), hl.struct(s='sample2')])
        ht = ht.annotate(entries=[hl.struct(GT=hl.call(0, 1)), hl.struct(GT=hl.call(1, 1))])
        mt = ht._unlocalize_entries('entries', 'cols', ['s'])
        mt = hl.split_multi_hts(mt, permit_shuffle=True)
        mt._force_count_rows()
        assert mt.alleles.collect() == [['A', 'C'], ['A', 'G'], ['A', 'T']]

    @qobtest
    def test_issue_4527(self):
        mt = hl.utils.range_matrix_table(1, 1)
        mt = mt.key_rows_by(locus=hl.locus(hl.str(mt.row_idx + 1), mt.row_idx + 1), alleles=['A', 'T'])
        mt = hl.split_multi(mt)
        self.assertEqual(1, mt._force_count_rows())

    @qobtest
    @test_timeout(batch=9 * 60)
    def test_ld_prune(self):
        r2_threshold = 0.001
        window_size = 5
        ds = hl.split_multi_hts(hl.import_vcf(resource('ldprune.vcf'), min_partitions=3))
        pruned_table = hl.ld_prune(ds.GT, r2=r2_threshold, bp_window_size=window_size)

        filtered_ds = ds.filter_rows(hl.is_defined(pruned_table[ds.row_key]))
        filtered_ds = filtered_ds.annotate_rows(stats=agg.stats(filtered_ds.GT.n_alt_alleles()))
        filtered_ds = filtered_ds.annotate_rows(mean=filtered_ds.stats.mean, sd_reciprocal=1 / filtered_ds.stats.stdev)

        n_samples = filtered_ds.count_cols()
        normalized_mean_imputed_genotype_expr = hl.if_else(
            hl.is_defined(filtered_ds['GT']),
            (filtered_ds['GT'].n_alt_alleles() - filtered_ds['mean'])
            * filtered_ds['sd_reciprocal']
            * (1 / hl.sqrt(n_samples)),
            0,
        )

        std_bm = BlockMatrix.from_entry_expr(normalized_mean_imputed_genotype_expr)

        self.assertEqual(std_bm.n_rows, 14)

        entries = ((std_bm @ std_bm.T) ** 2).entries()

        index_table = filtered_ds.add_row_index().rows().key_by('row_idx').select('locus')
        entries = entries.annotate(locus_i=index_table[entries.i].locus, locus_j=index_table[entries.j].locus)

        bad_pair = (
            (entries.entry >= r2_threshold)
            & (entries.locus_i.contig == entries.locus_j.contig)
            & (hl.abs(entries.locus_j.position - entries.locus_i.position) <= window_size)
            & (entries.i != entries.j)
        )

        self.assertEqual(entries.filter(bad_pair).count(), 0)

    def test_ld_prune_inputs(self):
        ds = hl.balding_nichols_model(n_populations=1, n_samples=1, n_variants=1)
        self.assertRaises(ValueError, lambda: hl.ld_prune(ds.GT, memory_per_core=0))
        self.assertRaises(ValueError, lambda: hl.ld_prune(ds.GT, bp_window_size=-1))
        self.assertRaises(ValueError, lambda: hl.ld_prune(ds.GT, r2=-1.0))
        self.assertRaises(ValueError, lambda: hl.ld_prune(ds.GT, r2=2.0))

    def test_ld_prune_no_prune(self):
        ds = hl.balding_nichols_model(n_populations=1, n_samples=10, n_variants=10, n_partitions=3)
        pruned_table = hl.ld_prune(ds.GT, r2=0.0, bp_window_size=0)
        expected_count = ds.filter_rows(agg.collect_as_set(ds.GT).size() > 1, keep=True).count_rows()
        self.assertEqual(pruned_table.count(), expected_count)

    def test_ld_prune_identical_variants(self):
        ds = hl.import_vcf(resource('ldprune2.vcf'), min_partitions=2)
        pruned_table = hl.ld_prune(ds.GT)
        self.assertEqual(pruned_table.count(), 1)

    @test_timeout(batch=5 * 60)
    def test_ld_prune_maf(self):
        ds = hl.balding_nichols_model(n_populations=1, n_samples=50, n_variants=10, n_partitions=10).cache()

        ht = ds.select_rows(p=hl.agg.sum(ds.GT.n_alt_alleles()) / (2 * 50)).rows()
        ht = ht.select(maf=hl.if_else(ht.p <= 0.5, ht.p, 1.0 - ht.p)).cache()

        pruned_table = hl.ld_prune(ds.GT, 0.0)
        positions = pruned_table.locus.position.collect()
        self.assertEqual(len(positions), 1)
        kept_position = hl.literal(positions[0])
        kept_maf = ht.filter(ht.locus.position == kept_position).maf.collect()[0]

        self.assertEqual(kept_maf, max(ht.maf.collect()))

    def test_ld_prune_call_expression(self):
        ds = hl.import_vcf(resource("ldprune2.vcf"), min_partitions=2)
        ds = ds.select_entries(foo=ds.GT)
        pruned_table = hl.ld_prune(ds.foo)
        self.assertEqual(pruned_table.count(), 1)

    def test_ld_prune_missing_entries(self):
        mt = hl.import_vcf(resource("ldprune2.vcf"), min_partitions=2).add_col_index()
        mt = mt.filter_entries(mt.col_idx > 1)
        result = hl.ld_prune(mt.GT)
        assert result.count() > 0

    @test_timeout(batch=5 * 60)
    def test_ld_prune_with_duplicate_row_keys(self):
        ds = hl.import_vcf(resource('ldprune2.vcf'), min_partitions=2)
        ds_duplicate = ds.annotate_rows(duplicate=[1, 2]).explode_rows('duplicate')
        pruned_table = hl.ld_prune(ds_duplicate.GT)
        self.assertEqual(pruned_table.count(), 1)

    def test_balding_nichols_model(self):
        hl.reset_global_randomness()
        ds = hl.balding_nichols_model(
            2,
            20,
            25,
            3,
            pop_dist=[1.0, 2.0],
            fst=[0.02, 0.06],
            af_dist=hl.rand_beta(a=0.01, b=2.0, lower=0.05, upper=0.95),
        )

        ds.entries().show(100, width=200)

        self.assertEqual(ds.count_cols(), 20)
        self.assertEqual(ds.count_rows(), 25)
        self.assertEqual(ds.n_partitions(), 3)

        glob = ds.globals
        self.assertEqual(hl.eval(glob.bn.n_populations), 2)
        self.assertEqual(hl.eval(glob.bn.n_samples), 20)
        self.assertEqual(hl.eval(glob.bn.n_variants), 25)
        self.assertEqual(hl.eval(glob.bn.pop_dist), [1, 2])
        self.assertEqual(hl.eval(glob.bn.fst), [0.02, 0.06])

    def test_balding_nichols_model_same_results(self):
        for mixture in [True, False]:
            hl.reset_global_randomness()
            ds1 = hl.balding_nichols_model(
                2,
                20,
                25,
                3,
                pop_dist=[1.0, 2.0],
                fst=[0.02, 0.06],
                af_dist=hl.rand_beta(a=0.01, b=2.0, lower=0.05, upper=0.95),
                mixture=mixture,
            )
            hl.reset_global_randomness()
            ds2 = hl.balding_nichols_model(
                2,
                20,
                25,
                3,
                pop_dist=[1.0, 2.0],
                fst=[0.02, 0.06],
                af_dist=hl.rand_beta(a=0.01, b=2.0, lower=0.05, upper=0.95),
                mixture=mixture,
            )
            self.assertTrue(ds1._same(ds2))

    def test_balding_nichols_model_af_ranges(self):
        def test_af_range(rand_func, min, max, seed):
            hl.reset_global_randomness()
            bn = hl.balding_nichols_model(3, 400, 400, af_dist=rand_func)
            self.assertTrue(bn.aggregate_rows(hl.agg.all((bn.ancestral_af > min) & (bn.ancestral_af < max))))

        test_af_range(hl.rand_beta(0.01, 2, 0.2, 0.8), 0.2, 0.8, 0)
        test_af_range(hl.rand_beta(3, 3, 0.4, 0.6), 0.4, 0.6, 1)
        test_af_range(hl.rand_unif(0.4, 0.7), 0.4, 0.7, 2)
        test_af_range(hl.rand_beta(4, 6), 0, 1, 3)

    @test_timeout(batch=6 * 60)
    def test_balding_nichols_stats(self):
        def test_stat(k, n, m, seed):
            hl.reset_global_randomness()
            bn = hl.balding_nichols_model(k, n, m, af_dist=hl.rand_unif(0.1, 0.9))

            # test pop distribution
            pop_counts = bn.aggregate_cols(hl.agg.group_by(bn.pop, hl.agg.count()))
            for i, count in pop_counts.items():
                self.assertAlmostEqual(count / n, 1 / k, delta=0.1 * n / k)

            # test af distribution
            def variance(expr):
                return hl.bind(lambda mean: hl.mean(hl.map(lambda elt: (elt - mean) ** 2, expr)), hl.mean(expr))

            delta_mean = 0.2  # consider alternatives to 0.2
            delta_var = 0.1
            per_row = hl.bind(
                lambda mean, var, ancestral: (ancestral > mean - delta_mean)
                & (ancestral < mean + delta_mean)
                & (0.1 * ancestral * (1 - ancestral) > var - delta_var)
                & (0.1 * ancestral * (1 - ancestral) < var + delta_var),
                hl.mean(bn.af),
                variance(bn.af),
                bn.ancestral_af,
            )
            self.assertTrue(bn.aggregate_rows(hl.agg.all(per_row)))

            # test genotype distribution
            stats_gt_by_pop = hl.agg.group_by(bn.pop, hl.agg.stats(hl.float(bn.GT.n_alt_alleles()))).values()
            bn = bn.select_rows(
                sum_af=hl.sum(bn.af), sum_mean_gt_by_pop=hl.sum(hl.map(lambda x: x.mean, stats_gt_by_pop))
            )
            sum_af = bn.aggregate_rows(hl.agg.sum(bn.sum_af))
            sum_mean_gt = bn.aggregate_rows(hl.agg.sum(bn.sum_mean_gt_by_pop))
            self.assertAlmostEqual(sum_mean_gt, 2 * sum_af, delta=0.1 * m * k)

        test_stat(10, 100, 100, 0)
        test_stat(40, 400, 20, 12)

    @skip_when_service_backend(reason='flaky, incorrect alleles in output')
    def test_balding_nichols_model_phased(self):
        bn_ds = hl.balding_nichols_model(1, 5, 5, phased=True)
        assert bn_ds.aggregate_entries(hl.agg.all(bn_ds.GT.phased)) is True
        actual = bn_ds.GT.collect()
        self.assertListEqual(
            [c.alleles for c in actual],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 0],
                [1, 1],
                [1, 1],
                [0, 1],
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 1],
                [1, 1],
                [0, 1],
                [1, 1],
                [1, 1],
            ],
        )

    def test_de_novo(self):
        mt = hl.import_vcf(resource('denovo.vcf'))
        mt = mt.filter_rows(mt.locus.in_y_par(), keep=False)  # de_novo_finder doesn't know about y PAR
        ped = hl.Pedigree.read(resource('denovo.fam'))
        r = hl.de_novo(mt, ped, mt.info.ESP)
        r = r.select(
            prior=r.prior,
            kid_id=r.proband.s,
            dad_id=r.father.s,
            mom_id=r.mother.s,
            p_de_novo=r.p_de_novo,
            confidence=r.confidence,
        ).key_by('locus', 'alleles', 'kid_id', 'dad_id', 'mom_id')

        truth = hl.import_table(resource('denovo.out'), impute=True, comment='#')
        truth = truth.select(
            locus=hl.locus(truth['Chr'], truth['Pos']),
            alleles=[truth['Ref'], truth['Alt']],
            kid_id=truth['Child_ID'],
            dad_id=truth['Dad_ID'],
            mom_id=truth['Mom_ID'],
            p_de_novo=truth['Prob_dn'],
            confidence=truth['Validation_Likelihood'].split('_')[0],
        ).key_by('locus', 'alleles', 'kid_id', 'dad_id', 'mom_id')

        j = r.join(truth, how='outer')
        self.assertTrue(j.all((j.confidence == j.confidence_1) & (hl.abs(j.p_de_novo - j.p_de_novo_1) < 1e-4)))

    def test_de_novo_error(self):
        mt = hl.import_vcf(resource('denovo.vcf'))
        ped = hl.Pedigree.read(resource('denovo.fam'))

        with pytest.raises(Exception, match='pop_frequency_prior'):
            hl.de_novo(mt, ped, pop_frequency_prior=2.0).count()

    def test_de_novo_ignore_computed_af_runs(self):
        mt = hl.import_vcf(resource('denovo.vcf'))
        ped = hl.Pedigree.read(resource('denovo.fam'))

        hl.de_novo(mt, ped, pop_frequency_prior=mt.info.ESP, ignore_in_sample_allele_frequency=True).count()

    def test_warn_if_no_intercept(self):
        mt = hl.balding_nichols_model(1, 1, 1).add_row_index().add_col_index()
        intercept = hl.float64(1.0)

        for covariates in [
            [],
            [mt.row_idx],
            [mt.col_idx],
            [mt.GT.n_alt_alleles()],
            [mt.row_idx, mt.col_idx, mt.GT.n_alt_alleles()],
        ]:
            self.assertTrue(hl.methods.statgen._warn_if_no_intercept('', covariates))
            self.assertFalse(hl.methods.statgen._warn_if_no_intercept('', [intercept, *covariates]))

    def test_regression_field_dependence(self):
        mt = hl.utils.range_matrix_table(10, 10)
        mt = mt.annotate_cols(c1=hl.literal([x % 2 == 0 for x in range(10)])[mt.col_idx], c2=hl.rand_norm(0, 1))
        mt = mt.annotate_entries(e1=hl.int(hl.rand_norm(0, 1) * 10))

        x_expr = hl.case().when(mt.c2 < 0, 0).default(mt.e1)

        hl.logistic_regression_rows('wald', y=mt.c1, x=x_expr, covariates=[1])
        hl.poisson_regression_rows('wald', y=mt.c1, x=x_expr, covariates=[1])
        hl.linear_regression_rows(y=mt.c1, x=x_expr, covariates=[1])


@pytest.fixture
def logistic_epacts_mt():
    # 2535 samples from 1K Genomes Project
    # Locus("22", 16060511)  # MAC  623
    # Locus("22", 16115878)  # MAC  370
    # Locus("22", 16115882)  # MAC 1207
    # Locus("22", 16117940)  # MAC    7
    # Locus("22", 16117953)  # MAC   21
    covariates = hl.import_table(
        resource('regressionLogisticEpacts.cov'), key='IND_ID', types={'PC1': hl.tfloat, 'PC2': hl.tfloat}
    )
    fam = hl.import_fam(resource('regressionLogisticEpacts.fam'))

    mt = hl.import_vcf(resource('regressionLogisticEpacts.vcf'))
    mt = mt.annotate_cols(**covariates[mt.s], **fam[mt.s])
    return mt


def test_logistic_regression_epacts_wald(logistic_epacts_mt):
    mt = logistic_epacts_mt
    actual = hl.logistic_regression_rows(
        test='wald', y=mt.is_case, x=mt.GT.n_alt_alleles(), covariates=[1.0, mt.is_female, mt.PC1, mt.PC2]
    ).collect()

    assert actual[0].locus == hl.Locus("22", 16060511, 'GRCh37')
    assert actual[0].beta == pytest.approx(-0.097476, rel=1e-4)
    assert actual[0].standard_error == pytest.approx(0.087478, rel=1e-4)
    assert actual[0].z_stat == pytest.approx(-1.1143, rel=1e-4)
    assert actual[0].p_value == pytest.approx(0.26516, rel=1e-4)

    assert actual[1].locus == hl.Locus("22", 16115878, 'GRCh37')
    assert actual[1].beta == pytest.approx(-0.052632, rel=1e-4)
    assert actual[1].standard_error == pytest.approx(0.11272, rel=1e-4)
    assert actual[1].z_stat == pytest.approx(-0.46691, rel=1e-4)
    assert actual[1].p_value == pytest.approx(0.64056, rel=1e-4)

    assert actual[2].locus == hl.Locus("22", 16115882, 'GRCh37')
    assert actual[2].beta == pytest.approx(-0.15598, rel=1e-4)
    assert actual[2].standard_error == pytest.approx(0.079508, rel=1e-4)
    assert actual[2].z_stat == pytest.approx(-1.9619, rel=1e-4)
    assert actual[2].p_value == pytest.approx(0.049779, rel=1e-4)

    assert actual[3].locus == hl.Locus("22", 16117940, 'GRCh37')
    assert actual[3].beta == pytest.approx(-0.88059, rel=1e-4)
    assert actual[3].standard_error == pytest.approx(0.83769, rel=1e-2)
    assert actual[3].z_stat == pytest.approx(-1.0512, rel=1e-2)
    assert actual[3].p_value == pytest.approx(0.29316, rel=1e-2)

    assert actual[4].locus == hl.Locus("22", 16117953, 'GRCh37')
    assert actual[4].beta == pytest.approx(0.54921, rel=1e-4)
    assert actual[4].standard_error == pytest.approx(0.4517, rel=1e-3)
    assert actual[4].z_stat == pytest.approx(1.2159, rel=1e-3)
    assert actual[4].p_value == pytest.approx(0.22403, rel=1e-3)


def test_logistic_regression_epacts_lrt(logistic_epacts_mt):
    mt = logistic_epacts_mt
    actual = hl.logistic_regression_rows(
        test='lrt', y=mt.is_case, x=mt.GT.n_alt_alleles(), covariates=[1.0, mt.is_female, mt.PC1, mt.PC2]
    ).collect()

    assert actual[0].locus == hl.Locus("22", 16060511, 'GRCh37')
    assert actual[0].p_value == pytest.approx(0.26475, rel=1e-4)

    assert actual[1].locus == hl.Locus("22", 16115878, 'GRCh37')
    assert actual[1].p_value == pytest.approx(0.64046, rel=1e-4)

    assert actual[2].locus == hl.Locus("22", 16115882, 'GRCh37')
    assert actual[2].p_value == pytest.approx(0.049675, rel=1e-4)

    assert actual[3].locus == hl.Locus("22", 16117940, 'GRCh37')
    assert actual[3].p_value == pytest.approx(0.26984, rel=1e-4)

    assert actual[4].locus == hl.Locus("22", 16117953, 'GRCh37')
    assert actual[4].p_value == pytest.approx(0.21692, rel=1e-4)


def test_logistic_regression_epacts_score(logistic_epacts_mt):
    # The name of this test suggests it was originally a comparison to EPACTS. The original EPACTS
    # values were slightly different from the output of lowered logistic regression. I regenerated
    # this test's expected values using R.
    #
    # 1. Export the data into an R-friendly format:
    #
    #     mt = logistic_epacts_mt()
    #     mt = mt.select_cols(
    #         y=hl.int32(mt.is_case),
    #         c1=1.0,
    #         c2=hl.int32(mt.is_female),
    #         c3=mt.PC1,
    #         c4=mt.PC2,
    #         x=hl.agg.collect(mt.GT.n_alt_alleles())
    #     )
    #     mt = mt.transmute_cols(**{
    #         f'x{i}': mt.x[i] for i in range(mt.count_rows())
    #     })
    #     mt.cols().export('phenos.tsv')
    #
    # 2. Run this model repeatedly for each x:
    #
    #     df = read.table(file = 'phenos.csv', sep = '\t', header = TRUE)
    #     poisfit <- glm(df$y ~ df$c1 + df$c2 + df$c3 + df$c4 + df$x0, family="binomial")
    #     poisfitnull <- glm(df$y ~ df$c1 + df$c2 + df$c3 + df$c4, family="binomial")
    #     scoretest <- anova(poisfitnull, poisfit, test="Rao")
    #     chi2 <- scoretest[["Rao"]][2]
    #     pval <- scoretest[["Pr(>Chi)"]][2]
    #
    mt = logistic_epacts_mt
    actual = hl.logistic_regression_rows(
        test='score', y=mt.is_case, x=mt.GT.n_alt_alleles(), covariates=[1.0, mt.is_female, mt.PC1, mt.PC2]
    ).collect()

    assert actual[0].locus == hl.Locus("22", 16060511, 'GRCh37')
    assert actual[0].chi_sq_stat == pytest.approx(1.242482, rel=1e-5)
    assert actual[0].p_value == pytest.approx(0.2649933, rel=1e-5)

    assert actual[1].locus == hl.Locus("22", 16115878, 'GRCh37')
    assert actual[1].chi_sq_stat == pytest.approx(0.218038, rel=1e-5)
    assert actual[1].p_value == pytest.approx(0.6405389, rel=1e-5)

    assert actual[2].locus == hl.Locus("22", 16115882, 'GRCh37')
    assert actual[2].chi_sq_stat == pytest.approx(3.850985, rel=1e-5)
    assert actual[2].p_value == pytest.approx(0.04971679, rel=1e-5)

    assert actual[3].locus == hl.Locus("22", 16117940, 'GRCh37')
    assert actual[3].chi_sq_stat == pytest.approx(1.175474, rel=1e-5)
    assert actual[3].p_value == pytest.approx(0.2782793, rel=1e-5)

    assert actual[4].locus == hl.Locus("22", 16117953, 'GRCh37')
    assert actual[4].chi_sq_stat == pytest.approx(1.514245, rel=1e-5)
    assert actual[4].p_value == pytest.approx(0.2184924, rel=1e-5)


def test_logistic_regression_epacts_firth(logistic_epacts_mt):
    mt = logistic_epacts_mt
    actual = hl.logistic_regression_rows(
        test='firth', y=mt.is_case, x=mt.GT.n_alt_alleles(), covariates=[1.0, mt.is_female, mt.PC1, mt.PC2]
    ).collect()

    assert actual[0].locus == hl.Locus("22", 16060511, 'GRCh37')
    assert actual[0].beta == pytest.approx(-0.097079, rel=1e-4)
    assert actual[0].p_value == pytest.approx(0.26593, rel=1e-4)

    assert actual[1].locus == hl.Locus("22", 16115878, 'GRCh37')
    assert actual[1].beta == pytest.approx(-0.052301, rel=1e-4)
    assert actual[1].p_value == pytest.approx(0.64197, rel=1e-4)

    assert actual[2].locus == hl.Locus("22", 16115882, 'GRCh37')
    assert actual[2].beta == pytest.approx(-0.15567, rel=1e-4)
    assert actual[2].p_value == pytest.approx(0.04991, rel=1e-4)

    assert actual[3].locus == hl.Locus("22", 16117940, 'GRCh37')
    assert actual[3].beta == pytest.approx(-0.7524, rel=1e-4)
    assert actual[3].p_value == pytest.approx(0.30731, rel=1e-4)

    assert actual[4].locus == hl.Locus("22", 16117953, 'GRCh37')
    assert actual[4].beta == pytest.approx(0.5258, rel=1e-4)
    assert actual[4].p_value == pytest.approx(0.22562, rel=1e-4)


## issue 13788
def test_logistic_regression_y_parameter_sanity():
    mt = hl.utils.range_matrix_table(2, 2)
    mt = mt.annotate_entries(prod=mt.row_idx * mt.col_idx)

    with pytest.raises(hl.ExpressionException):
        hl.logistic_regression_rows(test='wald', x=mt.prod, y=mt.row_idx, covariates=[1.0]).describe()
