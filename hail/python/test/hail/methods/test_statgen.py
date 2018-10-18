import unittest
from subprocess import DEVNULL, call as syscall
import numpy as np
import pytest

import hail as hl
import hail.expr.aggregators as agg
import hail.utils as utils
from hail.linalg import BlockMatrix
from ..helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
    def test_ibd(self):
        dataset = get_dataset()

        def plinkify(ds, min=None, max=None):
            vcf = utils.new_temp_file(prefix="plink", suffix="vcf")
            plinkpath = utils.new_temp_file(prefix="plink")
            hl.export_vcf(ds, vcf)
            threshold_string = "{} {}".format("--min {}".format(min) if min else "",
                                              "--max {}".format(max) if max else "")

            plink_command = "plink --double-id --allow-extra-chr --vcf {} --genome full --out {} {}" \
                .format(utils.uri_path(vcf),
                        utils.uri_path(plinkpath),
                        threshold_string)
            result_file = utils.uri_path(plinkpath + ".genome")

            syscall(plink_command, shell=True, stdout=DEVNULL, stderr=DEVNULL)

            ### format of .genome file is:
            # _, fid1, iid1, fid2, iid2, rt, ez, z0, z1, z2, pihat, phe,
            # dst, ppc, ratio, ibs0, ibs1, ibs2, homhom, hethet (+ separated)

            ### format of ibd is:
            # i (iid1), j (iid2), ibd: {Z0, Z1, Z2, PI_HAT}, ibs0, ibs1, ibs2
            results = {}
            with open(result_file) as f:
                f.readline()
                for line in f:
                    row = line.strip().split()
                    results[(row[1], row[3])] = (list(map(float, row[6:10])),
                                                 list(map(int, row[14:17])))
            return results

        def compare(ds, min=None, max=None):
            plink_results = plinkify(ds, min, max)
            hail_results = hl.identity_by_descent(ds, min=min, max=max).collect()

            for row in hail_results:
                key = (row.i, row.j)
                self.assertAlmostEqual(plink_results[key][0][0], row.ibd.Z0, places=4)
                self.assertAlmostEqual(plink_results[key][0][1], row.ibd.Z1, places=4)
                self.assertAlmostEqual(plink_results[key][0][2], row.ibd.Z2, places=4)
                self.assertAlmostEqual(plink_results[key][0][3], row.ibd.PI_HAT, places=4)
                self.assertEqual(plink_results[key][1][0], row.ibs0)
                self.assertEqual(plink_results[key][1][1], row.ibs1)
                self.assertEqual(plink_results[key][1][2], row.ibs2)

        compare(dataset)
        compare(dataset, min=0.0, max=1.0)
        dataset = dataset.annotate_rows(dummy_maf=0.01)
        hl.identity_by_descent(dataset, dataset['dummy_maf'], min=0.0, max=1.0)
        hl.identity_by_descent(dataset, hl.float32(dataset['dummy_maf']), min=0.0, max=1.0)

    def test_impute_sex_same_as_plink(self):
        ds = hl.import_vcf(resource('x-chromosome.vcf'))

        sex = hl.impute_sex(ds.GT, include_par=True)

        vcf_file = utils.uri_path(utils.new_temp_file(prefix="plink", suffix="vcf"))
        out_file = utils.uri_path(utils.new_temp_file(prefix="plink"))

        hl.export_vcf(ds, vcf_file)

        utils.run_command(["plink", "--vcf", vcf_file, "--const-fid",
                           "--check-sex", "--silent", "--out", out_file])

        plink_sex = hl.import_table(out_file + '.sexcheck',
                                    delimiter=' +',
                                    types={'SNPSEX': hl.tint32,
                                           'F': hl.tfloat64})
        plink_sex = plink_sex.select('IID', 'SNPSEX', 'F')
        plink_sex = plink_sex.select(
            s=plink_sex.IID,
            is_female=hl.cond(plink_sex.SNPSEX == 2,
                              True,
                              hl.cond(plink_sex.SNPSEX == 1,
                                      False,
                                      hl.null(hl.tbool))),
            f_stat=plink_sex.F).key_by('s')

        sex = sex.select('is_female', 'f_stat')

        self.assertTrue(plink_sex._same(sex.select_globals(), tolerance=1e-3))

        ds = ds.annotate_rows(aaf=(agg.call_stats(ds.GT, ds.alleles)).AF[1])

        self.assertTrue(hl.impute_sex(ds.GT)._same(hl.impute_sex(ds.GT, aaf='aaf')))

    def test_linreg(self):
        phenos = hl.import_table(resource('regressionLinear.pheno'),
                                 types={'Pheno': hl.tfloat64},
                                 key='Sample')
        covs = hl.import_table(resource('regressionLinear.cov'),
                               types={'Cov1': hl.tfloat64, 'Cov2': hl.tfloat64},
                               key='Sample')

        mt = hl.import_vcf(resource('regressionLinear.vcf'))
        mt = mt.annotate_cols(pheno=phenos[mt.s].Pheno, cov=covs[mt.s])
        mt = mt.annotate_entries(x=mt.GT.n_alt_alleles()).cache()

        t1 = hl.linear_regression_rows(
            y=mt.pheno, x=mt.GT.n_alt_alleles(), covariates=[1.0, mt.cov.Cov1, mt.cov.Cov2 + 1 - 1])
        t1 = t1.select(p=t1.p_value)

        t2 = hl.linear_regression_rows(
            y=mt.pheno, x=mt.x, covariates=[1.0, mt.cov.Cov1, mt.cov.Cov2])
        t2 = t2.select(p=t2.p_value)

        t3 = hl.linear_regression_rows(
            y=[mt.pheno], x=mt.x, covariates=[1.0, mt.cov.Cov1, mt.cov.Cov2])
        t3 = t3.select(p=t3.p_value[0])

        t4 = hl.linear_regression_rows(
            y=[mt.pheno, mt.pheno], x=mt.x, covariates=[1.0, mt.cov.Cov1, mt.cov.Cov2])
        t4a = t4.select(p=t4.p_value[0])
        t4b = t4.select(p=t4.p_value[1])

        self.assertTrue(t1._same(t2))
        self.assertTrue(t1._same(t3))
        self.assertTrue(t1._same(t4a))
        self.assertTrue(t1._same(t4b))

    def test_linreg_pass_through(self):
        phenos = hl.import_table(resource('regressionLinear.pheno'),
                                 types={'Pheno': hl.tfloat64},
                                 key='Sample')
        covs = hl.import_table(resource('regressionLinear.cov'),
                               types={'Cov1': hl.tfloat64, 'Cov2': hl.tfloat64},
                               key='Sample')

        mt = hl.import_vcf(resource('regressionLinear.vcf')).annotate_rows(foo = hl.struct(bar=hl.rand_norm(0, 1)))

        # single group
        lr_result = hl.linear_regression_rows(phenos[mt.s].Pheno, mt.GT.n_alt_alleles(), [1.0],
                                              pass_through=['filters', mt.foo.bar, mt.qual])

        assert mt.aggregate_rows(hl.agg.all(mt.foo.bar == lr_result[mt.row_key].bar))

        # chained
        lr_result = hl.linear_regression_rows([[phenos[mt.s].Pheno]], mt.GT.n_alt_alleles(), [1.0],
                                              pass_through=['filters', mt.foo.bar, mt.qual])

        assert mt.aggregate_rows(hl.agg.all(mt.foo.bar == lr_result[mt.row_key].bar))

        # check types
        assert 'filters' in lr_result.row
        assert lr_result.filters.dtype == mt.filters.dtype

        assert 'bar' in lr_result.row
        assert lr_result.bar.dtype == mt.foo.bar.dtype

        assert 'qual' in lr_result.row
        assert lr_result.qual.dtype == mt.qual.dtype

        # should run successfully with key fields
        hl.linear_regression_rows([[phenos[mt.s].Pheno]], mt.GT.n_alt_alleles(), [1.0],
                                  pass_through=['locus', 'alleles'])

        # complex expression
        with pytest.raises(ValueError):
            hl.linear_regression_rows([[phenos[mt.s].Pheno]], mt.GT.n_alt_alleles(), [1.0],
                                      pass_through=[mt.filters.length()])


    def test_linreg_chained(self):
        phenos = hl.import_table(resource('regressionLinear.pheno'),
                                 types={'Pheno': hl.tfloat64},
                                 key='Sample')
        covs = hl.import_table(resource('regressionLinear.cov'),
                               types={'Cov1': hl.tfloat64, 'Cov2': hl.tfloat64},
                               key='Sample')

        mt = hl.import_vcf(resource('regressionLinear.vcf'))
        mt = mt.annotate_cols(pheno=phenos[mt.s].Pheno, cov=covs[mt.s])
        mt = mt.annotate_entries(x=mt.GT.n_alt_alleles()).cache()

        t1 = hl.linear_regression_rows(y=[[mt.pheno], [mt.pheno]], x=mt.x, covariates=[1, mt.cov.Cov1, mt.cov.Cov2])
        def all_eq(*args):
            pred = True
            for a in args:
                if isinstance(a, hl.expr.Expression) \
                        and isinstance(a.dtype, hl.tarray) \
                        and isinstance(a.dtype.element_type, hl.tarray):
                    pred = pred & (hl.all(lambda x: x,
                        hl.map(lambda elt: ((hl.is_nan(elt[0]) & hl.is_nan(elt[1])) | (elt[0] == elt[1])),
                               hl.zip(a[0], a[1]))))
                else:
                    pred = pred & ((hl.is_nan(a[0]) & hl.is_nan(a[1])) | (a[0] == a[1]))
            return pred

        assert t1.aggregate(hl.agg.all(
            all_eq(t1.n,
                   t1.sum_x,
                   t1.y_transpose_x,
                   t1.beta,
                   t1.standard_error,
                   t1.t_stat,
                   t1.p_value)))

        mt2 = mt.filter_cols(mt.cov.Cov2 >= 0)
        mt3 = mt.filter_cols(mt.cov.Cov2 <= 0)

        # test that chained linear regression can replicate separate calls with different missingness
        t2 = hl.linear_regression_rows(y=mt2.pheno, x=mt2.x, covariates=[1, mt2.cov.Cov1])
        t3 = hl.linear_regression_rows(y=mt3.pheno, x=mt3.x, covariates=[1, mt3.cov.Cov1])

        chained = hl.linear_regression_rows(y=[[hl.case().when(mt.cov.Cov2 >= 0, mt.pheno).or_missing()],
                                               [hl.case().when(mt.cov.Cov2 <= 0, mt.pheno).or_missing()]],
                                            x=mt.x,
                                            covariates=[1, mt.cov.Cov1])
        chained = chained.annotate(r0=t2[chained.key], r1=t3[chained.key])
        assert chained.aggregate(hl.agg.all(
            all_eq([chained.n[0], chained.r0.n],
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
                   [chained.p_value[1][0], chained.r1.p_value])))

        # test differential missingness against each other
        phenos = [hl.case().when(mt.cov.Cov2 >= -1, mt.pheno).or_missing(),
                  hl.case().when(mt.cov.Cov2 <= 1, mt.pheno).or_missing()]
        t4 = hl.linear_regression_rows(phenos, mt.x, covariates=[1])
        t5 = hl.linear_regression_rows([phenos], mt.x, covariates=[1])

        t5 = t5.annotate(**{x: t5[x][0] for x in ['n', 'sum_x', 'y_transpose_x', 'beta', 'standard_error', 't_stat', 'p_value']})
        assert t4._same(t5)


    def test_linear_regression_without_intercept(self):
        pheno = hl.import_table(resource('regressionLinear.pheno'),
                                key='Sample',
                                missing='0',
                                types={'Pheno': hl.tfloat})
        mt = hl.import_vcf(resource('regressionLinear.vcf'))
        ht = hl.linear_regression_rows(y=pheno[mt.s].Pheno,
                                       x=mt.GT.n_alt_alleles(),
                                       covariates=[])
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
    def test_linear_regression_with_cov(self):

        covariates = hl.import_table(resource('regressionLinear.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLinear.pheno'),
                                key='Sample',
                                missing='0',
                                types={'Pheno': hl.tfloat})

        mt = hl.import_vcf(resource('regressionLinear.vcf'))
        ht = hl.linear_regression_rows(y=pheno[mt.s].Pheno,
                                       x=mt.GT.n_alt_alleles(),
                                       covariates=[1.0] + list(covariates[mt.s].values()))

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

        covariates = hl.import_table(resource('regressionLinear.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLinear.pheno'),
                                key='Sample',
                                missing='0',
                                types={'Pheno': hl.tfloat})

        mt = hl.import_vcf(resource('regressionLinear.vcf'))
        ht = hl.linear_regression_rows(y=pheno[mt.s].Pheno,
                                       x=hl.pl_dosage(mt.PL),
                                       covariates=[1.0] + list(covariates[mt.s].values()))

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

        covariates = hl.import_table(resource('regressionLinear.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLinear.pheno'),
                                key='Sample',
                                missing='0',
                                types={'Pheno': hl.tfloat})
        mt = hl.import_gen(resource('regressionLinear.gen'), sample_file=resource('regressionLinear.sample'))
        ht = hl.linear_regression_rows(y=pheno[mt.s].Pheno,
                                       x=hl.gp_dosage(mt.GP),
                                       covariates=[1.0] + list(covariates[mt.s].values()))

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

    def test_linear_regression_with_import_fam_boolean(self):
        covariates = hl.import_table(resource('regressionLinear.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        fam = hl.import_fam(resource('regressionLinear.fam'))
        mt = hl.import_vcf(resource('regressionLinear.vcf'))
        ht = hl.linear_regression_rows(y=fam[mt.s].is_case,
                                       x=mt.GT.n_alt_alleles(),
                                       covariates=[1.0] + list(covariates[mt.s].values()))

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
        covariates = hl.import_table(resource('regressionLinear.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        fam = hl.import_fam(resource('regressionLinear.fam'),
                            quant_pheno=True,
                            missing='0')
        mt = hl.import_vcf(resource('regressionLinear.vcf'))
        ht = hl.linear_regression_rows(y=fam[mt.s].quant_pheno,
                                       x=mt.GT.n_alt_alleles(),
                                       covariates=[1.0] + list(covariates[mt.s].values()))

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
        covariates = hl.import_table(resource('regressionLinear.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLinear.pheno'),
                                key='Sample',
                                missing='0',
                                types={'Pheno': hl.tfloat})

        mt = hl.import_vcf(resource('regressionLinear.vcf'))
        single = hl.linear_regression_rows(y=pheno[mt.s].Pheno,
                                           x=mt.GT.n_alt_alleles(),
                                                         covariates=list(covariates[mt.s].values()))
        multi = hl.linear_regression_rows(y=[pheno[mt.s].Pheno, pheno[mt.s].Pheno],
                                                        x=mt.GT.n_alt_alleles(),
                                                        covariates=list(covariates[mt.s].values()))

        def eq(x1, x2):
            return (hl.is_nan(x1) & hl.is_nan(x2)) | (hl.abs(x1 - x2) < 1e-4)

        combined = single.annotate(multi = multi[single.key])
        self.assertTrue(combined.aggregate(hl.agg.all(
            eq(combined.p_value, combined.multi.p_value[0]) &
            eq(combined.multi.p_value[0], combined.multi.p_value[1]))))

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
        covariates = hl.import_table(resource('regressionLogistic.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLogisticBoolean.pheno'),
                                key='Sample',
                                missing='0',
                                types={'isCase': hl.tbool})
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))
        ht = hl.logistic_regression_rows('wald',
                                         y=pheno[mt.s].isCase,
                                         x=mt.GT.n_alt_alleles(),
                                         covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2])

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

    def test_logistic_regression_wald_test_pl(self):
        covariates = hl.import_table(resource('regressionLogistic.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLogisticBoolean.pheno'),
                                key='Sample',
                                missing='0',
                                types={'isCase': hl.tbool})
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))
        ht = hl.logistic_regression_rows(
            test='wald',
            y=pheno[mt.s].isCase,
            x=hl.pl_dosage(mt.PL),
            covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2])

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
        covariates = hl.import_table(resource('regressionLogistic.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLogisticBoolean.pheno'),
                                key='Sample',
                                missing='0',
                                types={'isCase': hl.tbool})
        mt = hl.import_gen(resource('regressionLogistic.gen'),
                           sample_file=resource('regressionLogistic.sample'))
        ht = hl.logistic_regression_rows(
            test='wald',
            y=pheno[mt.s].isCase,
            x=hl.gp_dosage(mt.GP),
            covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2])

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
        covariates = hl.import_table(resource('regressionLogistic.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLogisticBoolean.pheno'),
                                key='Sample',
                                missing='0',
                                types={'isCase': hl.tbool})
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))
        ht = hl.logistic_regression_rows(
            test='lrt',
            y=pheno[mt.s].isCase,
            x=mt.GT.n_alt_alleles(),
            covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2])

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
        covariates = hl.import_table(resource('regressionLogistic.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLogisticBoolean.pheno'),
                                key='Sample',
                                missing='0',
                                types={'isCase': hl.tbool})
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))
        ht = hl.logistic_regression_rows(
            test='score',
            y=pheno[mt.s].isCase,
            x=mt.GT.n_alt_alleles(),
            covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2])

        results = dict(hl.tuple([ht.locus.position, ht.row]).collect())

        self.assertAlmostEqual(results[1].chi_sq_stat, 0.1502364955, places=6)
        self.assertAlmostEqual(results[1].p_value, 0.6983094571, places=6)

        self.assertAlmostEqual(results[2].chi_sq_stat, 0.1823600965, places=6)
        self.assertAlmostEqual(results[2].p_value, 0.6693528073, places=6)

        self.assertAlmostEqual(results[3].chi_sq_stat, 7.047367694, places=6)
        self.assertAlmostEqual(results[3].p_value, 0.007938182229, places=6)

        def is_constant(r):
            return r.chi_sq_stat is None or r.chi_sq_stat < 1e-6

        self.assertTrue(is_constant(results[6]))
        self.assertTrue(is_constant(results[7]))
        self.assertTrue(is_constant(results[8]))
        self.assertTrue(is_constant(results[9]))
        self.assertTrue(is_constant(results[10]))

    def test_logistic_regression_epacts(self):
        covariates = hl.import_table(resource('regressionLogisticEpacts.cov'),
                                     key='IND_ID',
                                     types={'PC1': hl.tfloat, 'PC2': hl.tfloat})
        fam = hl.import_fam(resource('regressionLogisticEpacts.fam'))

        mt = hl.import_vcf(resource('regressionLogisticEpacts.vcf'))
        mt = mt.annotate_cols(**covariates[mt.s], **fam[mt.s])

        def get_results(table):
            return dict(hl.tuple([table.locus.position, table.row]).collect())

        wald = get_results(hl.logistic_regression_rows(
            test='wald',
            y=mt.is_case,
            x=mt.GT.n_alt_alleles(),
            covariates=[1.0, mt.is_female, mt.PC1, mt.PC2]))
        lrt = get_results(hl.logistic_regression_rows(
            test='lrt',
            y=mt.is_case,
            x=mt.GT.n_alt_alleles(),
            covariates=[1.0, mt.is_female, mt.PC1, mt.PC2]))
        score = get_results(hl.logistic_regression_rows(
            test='score',
            y=mt.is_case,
            x=mt.GT.n_alt_alleles(),
            covariates=[1.0, mt.is_female, mt.PC1, mt.PC2]))
        firth = get_results(hl.logistic_regression_rows(
            test='firth',
            y=mt.is_case,
            x=mt.GT.n_alt_alleles(),
            covariates=[1.0, mt.is_female, mt.PC1, mt.PC2]))

        # 2535 samples from 1K Genomes Project
        # Locus("22", 16060511)  # MAC  623
        # Locus("22", 16115878)  # MAC  370
        # Locus("22", 16115882)  # MAC 1207
        # Locus("22", 16117940)  # MAC    7
        # Locus("22", 16117953)  # MAC   21

        self.assertAlmostEqual(wald[16060511].beta, -0.097476, places=4)
        self.assertAlmostEqual(wald[16060511].standard_error, 0.087478, places=4)
        self.assertAlmostEqual(wald[16060511].z_stat, -1.1143, places=4)
        self.assertAlmostEqual(wald[16060511].p_value, 0.26516, places=4)
        self.assertAlmostEqual(lrt[16060511].p_value, 0.26475, places=4)
        self.assertAlmostEqual(score[16060511].p_value, 0.26499, places=4)
        self.assertAlmostEqual(firth[16060511].beta, -0.097079, places=4)
        self.assertAlmostEqual(firth[16060511].p_value, 0.26593, places=4)

        self.assertAlmostEqual(wald[16115878].beta, -0.052632, places=4)
        self.assertAlmostEqual(wald[16115878].standard_error, 0.11272, places=4)
        self.assertAlmostEqual(wald[16115878].z_stat, -0.46691, places=4)
        self.assertAlmostEqual(wald[16115878].p_value, 0.64056, places=4)
        self.assertAlmostEqual(lrt[16115878].p_value, 0.64046, places=4)
        self.assertAlmostEqual(score[16115878].p_value, 0.64054, places=4)
        self.assertAlmostEqual(firth[16115878].beta, -0.052301, places=4)
        self.assertAlmostEqual(firth[16115878].p_value, 0.64197, places=4)

        self.assertAlmostEqual(wald[16115882].beta, -0.15598, places=4)
        self.assertAlmostEqual(wald[16115882].standard_error, 0.079508, places=4)
        self.assertAlmostEqual(wald[16115882].z_stat, -1.9619, places=4)
        self.assertAlmostEqual(wald[16115882].p_value, 0.049779, places=4)
        self.assertAlmostEqual(lrt[16115882].p_value, 0.049675, places=4)
        self.assertAlmostEqual(score[16115882].p_value, 0.049675, places=4)
        self.assertAlmostEqual(firth[16115882].beta, -0.15567, places=4)
        self.assertAlmostEqual(firth[16115882].p_value, 0.04991, places=4)

        self.assertAlmostEqual(wald[16117940].beta, -0.88059, places=4)
        self.assertAlmostEqual(wald[16117940].standard_error, 0.83769, places=2)
        self.assertAlmostEqual(wald[16117940].z_stat, -1.0512, places=2)
        self.assertAlmostEqual(wald[16117940].p_value, 0.29316, places=2)
        self.assertAlmostEqual(lrt[16117940].p_value, 0.26984, places=4)
        self.assertAlmostEqual(score[16117940].p_value, 0.27828, places=4)
        self.assertAlmostEqual(firth[16117940].beta, -0.7524, places=4)
        self.assertAlmostEqual(firth[16117940].p_value, 0.30731, places=4)

        self.assertAlmostEqual(wald[16117953].beta, 0.54921, places=4)
        self.assertAlmostEqual(wald[16117953].standard_error, 0.4517, places=3)
        self.assertAlmostEqual(wald[16117953].z_stat, 1.2159, places=3)
        self.assertAlmostEqual(wald[16117953].p_value, 0.22403, places=3)
        self.assertAlmostEqual(lrt[16117953].p_value, 0.21692, places=4)
        self.assertAlmostEqual(score[16117953].p_value, 0.21849, places=4)
        self.assertAlmostEqual(firth[16117953].beta, 0.5258, places=4)
        self.assertAlmostEqual(firth[16117953].p_value, 0.22562, places=4)

    def test_logreg_pass_through(self):
        covariates = hl.import_table(resource('regressionLogistic.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLogisticBoolean.pheno'),
                                key='Sample',
                                missing='0',
                                types={'isCase': hl.tbool})
        mt = hl.import_vcf(resource('regressionLogistic.vcf')).annotate_rows(foo = hl.struct(bar=hl.rand_norm(0, 1)))
        ht = hl.logistic_regression_rows('wald',
                                         y=pheno[mt.s].isCase,
                                         x=mt.GT.n_alt_alleles(),
                                         covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2],
                                         pass_through=['filters', mt.foo.bar, mt.qual])


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
        covariates = hl.import_table(resource('regressionLogistic.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionPoisson.pheno'),
                                key='Sample',
                                missing='-1',
                                types={'count': hl.tint32})
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))
        ht = hl.poisson_regression_rows(
            test='wald',
            y=pheno[mt.s].count,
            x=mt.GT.n_alt_alleles(),
            covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2])

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
    def test_poission_regression_lrt(self):
        covariates = hl.import_table(resource('regressionLogistic.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionPoisson.pheno'),
                                key='Sample',
                                missing='-1',
                                types={'count': hl.tint32})
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))
        ht = hl.poisson_regression_rows(
            test='lrt',
            y=pheno[mt.s].count,
            x=mt.GT.n_alt_alleles(),
            covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2])

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
    def test_poission_regression_score_test(self):
        covariates = hl.import_table(resource('regressionLogistic.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionPoisson.pheno'),
                                key='Sample',
                                missing='-1',
                                types={'count': hl.tint32})
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))
        ht = hl.poisson_regression_rows(
            test='score',
            y=pheno[mt.s].count,
            x=mt.GT.n_alt_alleles(),
            covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2])

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
        covariates = hl.import_table(resource('regressionLogistic.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionPoisson.pheno'),
                                key='Sample',
                                missing='-1',
                                types={'count': hl.tint32})
        mt = hl.import_vcf(resource('regressionLogistic.vcf')).annotate_rows(foo = hl.struct(bar=hl.rand_norm(0, 1)))
        ht = hl.poisson_regression_rows(
            test='wald',
            y=pheno[mt.s].count,
            x=mt.GT.n_alt_alleles(),
            covariates=[1.0, covariates[mt.s].Cov1, covariates[mt.s].Cov2],
            pass_through=['filters', mt.foo.bar, mt.qual])

        assert mt.aggregate_rows(hl.agg.all(mt.foo.bar == ht[mt.row_key].bar))


    def test_genetic_relatedness_matrix(self):
        n, m = 100, 200
        hl.set_global_seed(0)
        mt = hl.balding_nichols_model(3, n, m, fst=[.9, .9, .9], n_partitions=4)

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
        hl.set_global_seed(0)
        mt = hl.balding_nichols_model(3, n, m, fst=[.9, .9, .9], n_partitions=4)

        g = BlockMatrix.from_entry_expr(mt.GT.n_alt_alleles()).to_numpy().T
        g_std = self._filter_and_standardize_cols(g)
        m1 = g_std.shape[1]
        self.assertTrue(m1 < m)
        k = (g_std @ g_std.T) * (n / m1)

        rrm = hl.realized_relationship_matrix(mt.GT).to_numpy()
        self.assertTrue(np.allclose(k, rrm))

    def test_row_correlation_vs_hardcode(self):
        data = [{'v': '1:1:A:C', 's': '1', 'GT': hl.Call([0, 0])},
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
                {'v': '1:3:C:G', 's': '4', 'GT': hl.null(hl.tcall)}]
        ht = hl.Table.parallelize(data, hl.dtype('struct{v: str, s: str, GT: call}'))
        mt = ht.to_matrix_table(['v'], ['s'])

        actual = hl.row_correlation(mt.GT.n_alt_alleles()).to_numpy()
        expected = [[1., -0.85280287, 0.42640143], [-0.85280287,  1., -0.5], [0.42640143, -0.5, 1.]]

        self.assertTrue(np.allclose(actual, expected))

    def test_row_correlation_vs_numpy(self):
        n, m = 11, 10
        hl.set_global_seed(0)
        mt = hl.balding_nichols_model(3, n, m, fst=[.9, .9, .9], n_partitions=2)
        mt = mt.annotate_rows(sd=agg.stats(mt.GT.n_alt_alleles()).stdev)
        mt = mt.filter_rows(mt.sd > 1e-30)

        g = BlockMatrix.from_entry_expr(mt.GT.n_alt_alleles()).to_numpy().T
        g_std = self._filter_and_standardize_cols(g)
        l = g_std.T @ g_std

        cor = hl.row_correlation(mt.GT.n_alt_alleles()).to_numpy()

        self.assertTrue(cor.shape[0] > 5 and cor.shape[0] == cor.shape[1])
        self.assertTrue(np.allclose(l, cor))

    def test_ld_matrix(self):
        data = [{'v': '1:1:A:C',       'cm': 0.1, 's': 'a', 'GT': hl.Call([0, 0])},
                {'v': '1:1:A:C',       'cm': 0.1, 's': 'b', 'GT': hl.Call([0, 0])},
                {'v': '1:1:A:C',       'cm': 0.1, 's': 'c', 'GT': hl.Call([0, 1])},
                {'v': '1:1:A:C',       'cm': 0.1, 's': 'd', 'GT': hl.Call([1, 1])},
                {'v': '1:2000000:G:T', 'cm': 0.9, 's': 'a', 'GT': hl.Call([0, 1])},
                {'v': '1:2000000:G:T', 'cm': 0.9, 's': 'b', 'GT': hl.Call([1, 1])},
                {'v': '1:2000000:G:T', 'cm': 0.9, 's': 'c', 'GT': hl.Call([0, 1])},
                {'v': '1:2000000:G:T', 'cm': 0.9, 's': 'd', 'GT': hl.Call([0, 0])},
                {'v': '2:1:C:G',       'cm': 0.2, 's': 'a', 'GT': hl.Call([0, 1])},
                {'v': '2:1:C:G',       'cm': 0.2, 's': 'b', 'GT': hl.Call([0, 0])},
                {'v': '2:1:C:G',       'cm': 0.2, 's': 'c', 'GT': hl.Call([1, 1])},
                {'v': '2:1:C:G',       'cm': 0.2, 's': 'd', 'GT': hl.null(hl.tcall)}]
        ht = hl.Table.parallelize(data, hl.dtype('struct{v: str, s: str, cm: float64, GT: call}'))
        ht = ht.transmute(**hl.parse_variant(ht.v))
        mt = ht.to_matrix_table(row_key=['locus', 'alleles'], col_key=['s'], row_fields=['cm'])

        self.assertTrue(np.allclose(
            hl.ld_matrix(mt.GT.n_alt_alleles(), mt.locus, radius=1e6).to_numpy(),
            [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))

        self.assertTrue(np.allclose(
            hl.ld_matrix(mt.GT.n_alt_alleles(), mt.locus, radius=2e6).to_numpy(),
            [[1., -0.85280287, 0.], [-0.85280287, 1., 0.], [0., 0., 1.]]))

        self.assertTrue(np.allclose(
            hl.ld_matrix(mt.GT.n_alt_alleles(), mt.locus, radius=0.5, coord_expr=mt.cm).to_numpy(),
            [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))

        self.assertTrue(np.allclose(
            hl.ld_matrix(mt.GT.n_alt_alleles(), mt.locus, radius=1.0, coord_expr=mt.cm).to_numpy(),
            [[1., -0.85280287, 0.], [-0.85280287, 1., 0.], [0., 0., 1.]]))

    def test_hwe_normalized_pca(self):
        mt = hl.balding_nichols_model(3, 100, 50)
        eigenvalues, scores, loadings = hl.hwe_normalized_pca(mt.GT, k=2, compute_loadings=True)

        self.assertEqual(len(eigenvalues), 2)
        self.assertTrue(isinstance(scores, hl.Table))
        self.assertEqual(scores.count(), 100)
        self.assertTrue(isinstance(loadings, hl.Table))

        _, _, loadings = hl.hwe_normalized_pca(mt.GT, k=2, compute_loadings=False)
        self.assertEqual(loadings, None)

    def test_pca_against_numpy(self):
        mt = hl.import_vcf(resource('tiny_m.vcf'))
        mt = mt.filter_rows(hl.len(mt.alleles) == 2)
        mt = mt.annotate_rows(AC=hl.agg.sum(mt.GT.n_alt_alleles()),
                              n_called=hl.agg.count_where(hl.is_defined(mt.GT)))
        mt = mt.filter_rows((mt.AC > 0) & (mt.AC < 2 * mt.n_called)).persist()
        n_rows = mt.count_rows()

        def make_expr(mean):
            return hl.cond(hl.is_defined(mt.GT),
                           (mt.GT.n_alt_alleles() - mean) / hl.sqrt(mean * (2 - mean) * n_rows / 2),
                           0)

        eigen, scores, loadings = hl.pca(hl.bind(make_expr, mt.AC / mt.n_called), k=3, compute_loadings=True)
        hail_scores = scores.explode('scores').scores.collect()
        hail_loadings = loadings.explode('loadings').loadings.collect()

        self.assertEqual(len(eigen), 3)
        self.assertEqual(scores.count(), mt.count_cols())
        self.assertEqual(loadings.count(), n_rows)

        # compute PCA with numpy
        def normalize(a):
            ms = np.mean(a, axis=0, keepdims=True)
            return np.divide(np.subtract(a, ms), np.sqrt(2.0 * np.multiply(ms / 2.0, 1 - ms / 2.0) * a.shape[1]))

        g = np.pad(np.diag([1.0, 1, 2]), ((0, 1), (0, 0)), mode='constant')
        g[1, 0] = 1.0 / 3
        n = normalize(g)
        U, s, V = np.linalg.svd(n, full_matrices=0)
        np_scores = U.dot(np.diag(s)).flatten()
        np_loadings = V.transpose().flatten()
        np_eigenvalues = np.multiply(s, s).flatten()

        def check(hail_array, np_array):
            self.assertEqual(len(hail_array), len(np_array))
            for i, (left, right) in enumerate(zip(hail_array, np_array)):
                self.assertAlmostEqual(abs(left), abs(right),
                                       msg=f'mismatch at index {i}: hl={left}, np={right}',
                                       places=4)

        check(eigen, np_eigenvalues)
        check(hail_scores, np_scores)
        check(hail_loadings, np_loadings)

    def _R_pc_relate(self, mt, maf):
        plink_file = utils.uri_path(utils.new_temp_file())
        hl.export_plink(mt, plink_file, ind_id=hl.str(mt.col_key[0]))
        utils.run_command(["Rscript",
                           resource("is/hail/methods/runPcRelate.R"),
                           plink_file,
                           str(maf)])

        types = {
            'ID1': hl.tstr,
            'ID2': hl.tstr,
            'nsnp': hl.tfloat64,
            'kin': hl.tfloat64,
            'k0': hl.tfloat64,
            'k1': hl.tfloat64,
            'k2': hl.tfloat64
        }
        plink_kin = hl.import_table(plink_file + '.out',
                                    delimiter=' +',
                                    types=types)
        return plink_kin.select(i=hl.struct(sample_idx=plink_kin.ID1),
                                j=hl.struct(sample_idx=plink_kin.ID2),
                                kin=plink_kin.kin,
                                ibd0=plink_kin.k0,
                                ibd1=plink_kin.k1,
                                ibd2=plink_kin.k2).key_by('i', 'j')

    def test_pc_relate_on_balding_nichols_against_R_pc_relate(self):
        mt = hl.balding_nichols_model(3, 100, 1000)
        mt = mt.key_cols_by(sample_idx=hl.str(mt.sample_idx))
        hkin = hl.pc_relate(mt.GT, 0.00, k=2).cache()
        rkin = self._R_pc_relate(mt, 0.00).cache()

        self.assertTrue(rkin.select("kin")._same(hkin.select("kin"), tolerance=1e-3, absolute=True))
        self.assertTrue(rkin.select("ibd0")._same(hkin.select("ibd0"), tolerance=1.3e-2, absolute=True))
        self.assertTrue(rkin.select("ibd1")._same(hkin.select("ibd1"), tolerance=2.6e-2, absolute=True))
        self.assertTrue(rkin.select("ibd2")._same(hkin.select("ibd2"), tolerance=1.3e-2, absolute=True))

    def test_pcrelate_paths(self):
        mt = hl.balding_nichols_model(3, 50, 100)
        _, scores2, _ = hl.hwe_normalized_pca(mt.GT, k=2, compute_loadings=False)
        _, scores3, _ = hl.hwe_normalized_pca(mt.GT, k=3, compute_loadings=False)

        kin1 = hl.pc_relate(mt.GT, 0.10, k=2, statistics='kin', block_size=64)
        kin_s1 = hl.pc_relate(mt.GT, 0.10, scores_expr=scores2[mt.col_key].scores,
                              statistics='kin', block_size=32)

        kin2 = hl.pc_relate(mt.GT, 0.05, k=2, min_kinship=0.01, statistics='kin2', block_size=128).cache()
        kin_s2 = hl.pc_relate(mt.GT, 0.05, scores_expr=scores2[mt.col_key].scores, min_kinship=0.01,
                              statistics='kin2', block_size=16)

        kin3 = hl.pc_relate(mt.GT, 0.02, k=3, min_kinship=0.1, statistics='kin20', block_size=64).cache()
        kin_s3 = hl.pc_relate(mt.GT, 0.02, scores_expr=scores3[mt.col_key].scores, min_kinship=0.1,
                              statistics='kin20', block_size=32)

        kin4 = hl.pc_relate(mt.GT, 0.01, k=3, statistics='all', block_size=128)
        kin_s4 = hl.pc_relate(mt.GT, 0.01, scores_expr=scores3[mt.col_key].scores, statistics='all', block_size=16)

        self.assertTrue(kin1._same(kin_s1, tolerance=1e-4))
        self.assertTrue(kin2._same(kin_s2, tolerance=1e-4))
        self.assertTrue(kin3._same(kin_s3, tolerance=1e-4))
        self.assertTrue(kin4._same(kin_s4, tolerance=1e-4))

        self.assertTrue(kin1.count() == 50 * 49 / 2)

        self.assertTrue(kin2.count() > 0)
        self.assertTrue(kin2.filter(kin2.kin < 0.01).count() == 0)

        self.assertTrue(kin3.count() > 0)
        self.assertTrue(kin3.filter(kin3.kin < 0.1).count() == 0)

    def test_split_multi_hts(self):
        ds1 = hl.import_vcf(resource('split_test.vcf'))
        ds1 = hl.split_multi_hts(ds1)
        ds2 = hl.import_vcf(resource('split_test_b.vcf'))
        df = ds1.rows()
        self.assertTrue(df.all((df.locus.position == 1180) | df.was_split))
        ds1 = ds1.drop('was_split', 'a_index')
        self.assertTrue(ds1._same(ds2))

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

    def test_issue_4527(self):
        mt = hl.utils.range_matrix_table(1, 1)
        mt = mt.key_rows_by(locus=hl.locus(hl.str(mt.row_idx+1), mt.row_idx+1), alleles=['A', 'T'])
        mt = hl.split_multi(mt)
        self.assertEqual(1, mt._force_count_rows())

    def test_ld_prune(self):
        r2_threshold = 0.001
        window_size = 5
        ds = hl.split_multi_hts(hl.import_vcf(resource('ldprune.vcf'), min_partitions=3))
        pruned_table = hl.ld_prune(ds.GT, r2=r2_threshold, bp_window_size=window_size)

        filtered_ds = ds.filter_rows(hl.is_defined(pruned_table[ds.row_key]))
        filtered_ds = filtered_ds.annotate_rows(stats=agg.stats(filtered_ds.GT.n_alt_alleles()))
        filtered_ds = filtered_ds.annotate_rows(
            mean=filtered_ds.stats.mean, sd_reciprocal=1 / filtered_ds.stats.stdev)

        n_samples = filtered_ds.count_cols()
        normalized_mean_imputed_genotype_expr = (
            hl.cond(hl.is_defined(filtered_ds['GT']),
                    (filtered_ds['GT'].n_alt_alleles() - filtered_ds['mean'])
                    * filtered_ds['sd_reciprocal'] * (1 / hl.sqrt(n_samples)), 0))

        std_bm = BlockMatrix.from_entry_expr(normalized_mean_imputed_genotype_expr)

        self.assertEqual(std_bm.n_rows, 14)

        entries = ((std_bm @ std_bm.T) ** 2).entries()

        index_table = filtered_ds.add_row_index().rows().key_by('row_idx').select('locus')
        entries = entries.annotate(locus_i=index_table[entries.i].locus, locus_j=index_table[entries.j].locus)

        bad_pair = (
            (entries.entry >= r2_threshold) &
            (entries.locus_i.contig == entries.locus_j.contig) &
            (hl.abs(entries.locus_j.position - entries.locus_i.position) <= window_size) &
            (entries.i != entries.j)
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

    def test_ld_prune_maf(self):
        ds = hl.balding_nichols_model(n_populations=1, n_samples=50, n_variants=10, n_partitions=10).cache()

        ht = ds.select_rows(p=hl.agg.sum(ds.GT.n_alt_alleles()) / (2 * 50)).rows()
        ht = ht.select(maf=hl.cond(ht.p <= 0.5, ht.p, 1.0 - ht.p)).cache()

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

    def test_ld_prune_with_duplicate_row_keys(self):
        ds = hl.import_vcf(resource('ldprune2.vcf'), min_partitions=2)
        ds_duplicate = ds.annotate_rows(duplicate=[1, 2]).explode_rows('duplicate')
        pruned_table = hl.ld_prune(ds_duplicate.GT)
        self.assertEqual(pruned_table.count(), 1)

    def test_balding_nichols_model(self):
        hl.set_global_seed(1)
        ds = hl.balding_nichols_model(2, 20, 25, 3,
                                      pop_dist=[1.0, 2.0],
                                      fst=[.02, .06],
                                      af_dist=hl.rand_beta(a=0.01, b=2.0, lower=0.05, upper=0.95))

        ds.entries().show(100, width=200)

        self.assertEqual(ds.count_cols(), 20)
        self.assertEqual(ds.count_rows(), 25)
        self.assertEqual(ds.n_partitions(), 3)

        glob = ds.globals
        self.assertEqual(hl.eval(glob.bn.n_populations), 2)
        self.assertEqual(hl.eval(glob.bn.n_samples), 20)
        self.assertEqual(hl.eval(glob.bn.n_variants), 25)
        self.assertEqual(hl.eval(glob.bn.pop_dist), [1, 2])
        self.assertEqual(hl.eval(glob.bn.fst), [.02, .06])

    def test_balding_nichols_model_same_results(self):
        hl.set_global_seed(1)
        ds1 = hl.balding_nichols_model(2, 20, 25, 3,
                                       pop_dist=[1.0, 2.0],
                                       fst=[.02, .06],
                                       af_dist=hl.rand_beta(a=0.01, b=2.0, lower=0.05, upper=0.95))
        hl.set_global_seed(1)
        ds2 = hl.balding_nichols_model(2, 20, 25, 3,
                                       pop_dist=[1.0, 2.0],
                                       fst=[.02, .06],
                                       af_dist=hl.rand_beta(a=0.01, b=2.0, lower=0.05, upper=0.95))
        self.assertTrue(ds1._same(ds2))

    def test_balding_nichols_model_af_ranges(self):
        def test_af_range(rand_func, min, max, seed):
            hl.set_global_seed(seed)
            bn = hl.balding_nichols_model(3, 400, 400, af_dist=rand_func)
            self.assertTrue(
                bn.aggregate_rows(
                    hl.agg.all((bn.ancestral_af > min) &
                               (bn.ancestral_af < max))))

        test_af_range(hl.rand_beta(.01, 2, .2, .8), .2, .8, 0)
        test_af_range(hl.rand_beta(3, 3, .4, .6), .4, .6, 1)
        test_af_range(hl.rand_unif(.4, .7), .4, .7, 2)
        test_af_range(hl.rand_beta(4, 6), 0, 1, 3)

    def test_balding_nichols_stats(self):
        def test_stat(k, n, m, seed):
            hl.set_global_seed(seed)
            bn = hl.balding_nichols_model(k, n, m, af_dist=hl.rand_unif(0.1, 0.9))

            # test pop distribution
            pop_counts = bn.aggregate_cols(hl.agg.group_by(bn.pop, hl.agg.count()))
            for i, count in pop_counts.items():
                self.assertAlmostEqual(count / n, 1 / k, delta=0.1 * n / k)

            # test af distribution
            def variance(expr):
                return hl.bind(lambda mean: hl.mean(hl.map(lambda elt: (elt - mean) ** 2, expr)), hl.mean(expr))
            delta_mean = 0.2 # consider alternatives to 0.2
            delta_var = 0.1
            per_row = hl.bind(lambda mean, var, ancestral:
                              (ancestral > mean - delta_mean) &
                              (ancestral < mean + delta_mean) &
                              (.1 * ancestral * (1 - ancestral) > var - delta_var) &
                              (.1 * ancestral * (1 - ancestral) < var + delta_var),
                              hl.mean(bn.af),
                              variance(bn.af),
                              bn.ancestral_af)
            self.assertTrue(bn.aggregate_rows(hl.agg.all(per_row)))

            # test genotype distribution
            stats_gt_by_pop = hl.agg.group_by(bn.pop, hl.agg.stats(hl.float(bn.GT.n_alt_alleles()))).values()
            bn = bn.select_rows(sum_af=hl.sum(bn.af),
                                sum_mean_gt_by_pop=hl.sum(hl.map(lambda x: x.mean, stats_gt_by_pop)))
            sum_af = bn.aggregate_rows(hl.agg.sum(bn.sum_af))
            sum_mean_gt = bn.aggregate_rows(hl.agg.sum(bn.sum_mean_gt_by_pop))
            self.assertAlmostEqual(sum_mean_gt, 2 * sum_af, delta=0.1 * m * k)

        test_stat(10, 100, 100, 0)
        test_stat(40, 400, 20, 12)

    def test_skat(self):
        ds2 = hl.import_vcf(resource('sample2.vcf'))

        covariates = (hl.import_table(resource("skat.cov"), impute=True)
                      .key_by("Sample"))

        phenotypes = (hl.import_table(resource("skat.pheno"),
                                      types={"Pheno": hl.tfloat64},
                                      missing="0")
                      .key_by("Sample"))

        intervals = (hl.import_locus_intervals(resource("skat.interval_list")))

        weights = (hl.import_table(resource("skat.weights"),
                                   types={"locus": hl.tlocus(),
                                          "weight": hl.tfloat64})
                   .key_by("locus"))

        ds = hl.split_multi_hts(ds2)
        ds = ds.annotate_rows(gene=intervals[ds.locus],
                              weight=weights[ds.locus].weight)
        ds = ds.annotate_cols(pheno=phenotypes[ds.s].Pheno,
                              cov=covariates[ds.s])
        ds = ds.annotate_cols(pheno=hl.cond(ds.pheno == 1.0,
                                            False,
                                            hl.cond(ds.pheno == 2.0,
                                                    True,
                                                    hl.null(hl.tbool))))

        hl.skat(key_expr=ds.gene,
                weight_expr=ds.weight,
                y=ds.pheno,
                x=ds.GT.n_alt_alleles(),
                covariates=[],
                logistic=False)._force_count()

        hl.skat(key_expr=ds.gene,
                weight_expr=ds.weight,
                y=ds.pheno,
                x=ds.GT.n_alt_alleles(),
                covariates=[],
                logistic=True)._force_count()

        hl.skat(key_expr=ds.gene,
                weight_expr=ds.weight,
                y=ds.pheno,
                x=ds.GT.n_alt_alleles(),
                covariates=[1.0, ds.cov.Cov1, ds.cov.Cov2],
                logistic=False)._force_count()

        hl.skat(key_expr=ds.gene,
                weight_expr=ds.weight,
                y=ds.pheno,
                x=hl.pl_dosage(ds.PL),
                covariates=[1.0, ds.cov.Cov1, ds.cov.Cov2],
                logistic=True)._force_count()

    def test_de_novo(self):
        mt = hl.import_vcf(resource('denovo.vcf'))
        mt = mt.filter_rows(mt.locus.in_y_par(), keep=False)  # de_novo_finder doesn't know about y PAR
        ped = hl.Pedigree.read(resource('denovo.fam'))
        r = hl.de_novo(mt, ped, mt.info.ESP)
        r = r.select(
            prior = r.prior,
            kid_id=r.proband.s,
            dad_id=r.father.s,
            mom_id=r.mother.s,
            p_de_novo=r.p_de_novo,
            confidence=r.confidence).key_by('locus', 'alleles', 'kid_id', 'dad_id', 'mom_id')

        truth = hl.import_table(resource('denovo.out'), impute=True, comment='#')
        truth = truth.select(
            locus=hl.locus(truth['Chr'], truth['Pos']),
            alleles=[truth['Ref'], truth['Alt']],
            kid_id=truth['Child_ID'],
            dad_id=truth['Dad_ID'],
            mom_id=truth['Mom_ID'],
            p_de_novo=truth['Prob_dn'],
            confidence=truth['Validation_Likelihood'].split('_')[0]).key_by('locus', 'alleles', 'kid_id', 'dad_id', 'mom_id')

        j = r.join(truth, how='outer')
        self.assertTrue(j.all((j.confidence == j.confidence_1) & (hl.abs(j.p_de_novo - j.p_de_novo_1) < 1e-4)))

    def test_warn_if_no_intercept(self):
        mt = hl.balding_nichols_model(1, 1, 1).add_row_index().add_col_index()
        intercept = hl.float64(1.0)

        for covariates in [[],
                           [mt.row_idx],
                           [mt.col_idx],
                           [mt.GT.n_alt_alleles()],
                           [mt.row_idx, mt.col_idx, mt.GT.n_alt_alleles()]]:
            self.assertTrue(hl.methods.statgen._warn_if_no_intercept('', covariates))
            self.assertFalse(hl.methods.statgen._warn_if_no_intercept('', [intercept] + covariates))
