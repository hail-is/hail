import unittest

import hail as hl
import hail.expr.aggregators as agg
from subprocess import DEVNULL, call as syscall
import numpy as np
from struct import unpack
import hail.utils as utils
from hail.linalg import BlockMatrix
from math import sqrt
from .utils import resource, doctest_resource, startTestHailContext, stopTestHailContext

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
    _dataset = None

    def get_dataset(self):
        if Tests._dataset is None:
            Tests._dataset = hl.split_multi_hts(hl.import_vcf(resource('sample.vcf')))
        return Tests._dataset

    def test_ibd(self):
        dataset = self.get_dataset()

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
        mt = mt.annotate_entries(x = mt.GT.n_alt_alleles()).cache()

        t1 = hl.linear_regression(
            y=mt.pheno, x=mt.GT.n_alt_alleles(), covariates=[mt.cov.Cov1, mt.cov.Cov2 + 1 - 1]).rows()
        t1 = t1.select(p=t1.linreg.p_value)

        t2 = hl.linear_regression(
            y=mt.pheno, x=mt.x, covariates=[mt.cov.Cov1, mt.cov.Cov2]).rows()
        t2 = t2.select(p=t2.linreg.p_value)

        t3 = hl.linear_regression(
            y=[mt.pheno], x=mt.x, covariates=[mt.cov.Cov1, mt.cov.Cov2]).rows()
        t3 = t3.select(p=t3.linreg.p_value[0])

        t4 = hl.linear_regression(
            y=[mt.pheno, mt.pheno], x=mt.x, covariates=[mt.cov.Cov1, mt.cov.Cov2]).rows()
        t4a = t4.select(p=t4.linreg.p_value[0])
        t4b = t4.select(p=t4.linreg.p_value[1])

        self.assertTrue(t1._same(t2))
        self.assertTrue(t1._same(t3))
        self.assertTrue(t1._same(t4a))
        self.assertTrue(t1._same(t4b))

    def test_linear_regression_with_two_cov(self):

        covariates = hl.import_table(resource('regressionLinear.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLinear.pheno'),
                                key='Sample',
                                missing='0',
                                types={'Pheno': hl.tfloat})

        mt = hl.import_vcf(resource('regressionLinear.vcf'))
        mt = hl.linear_regression(y=pheno[mt.s].Pheno,
                                  x=mt.GT.n_alt_alleles(),
                                  covariates=list(covariates[mt.s].values()))

        results = dict(mt.aggregate_rows(hl.agg.collect((mt.locus.position, mt.linreg))))

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

    def test_linear_regression_with_two_cov_pl(self):

        covariates = hl.import_table(resource('regressionLinear.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLinear.pheno'),
                                key='Sample',
                                missing='0',
                                types={'Pheno': hl.tfloat})

        mt = hl.import_vcf(resource('regressionLinear.vcf'))
        mt = hl.linear_regression(y=pheno[mt.s].Pheno,
                                  x=hl.pl_dosage(mt.PL),
                                  covariates=list(covariates[mt.s].values()))

        results = dict(mt.aggregate_rows(hl.agg.collect((mt.locus.position, mt.linreg))))

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

    def test_linear_regression_with_two_cov_dosage(self):

        covariates = hl.import_table(resource('regressionLinear.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLinear.pheno'),
                                key='Sample',
                                missing='0',
                                types={'Pheno': hl.tfloat})
        mt = hl.import_gen(resource('regressionLinear.gen'), sample_file=resource('regressionLinear.sample'))
        mt = hl.linear_regression(y=pheno[mt.s].Pheno,
                                  x=hl.gp_dosage(mt.GP),
                                  covariates=list(covariates[mt.s].values()))

        results = dict(mt.aggregate_rows(hl.agg.collect((mt.locus.position, mt.linreg))))

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

    def test_linear_regression_with_no_cov(self):
        pheno = hl.import_table(resource('regressionLinear.pheno'),
                                key='Sample',
                                missing='0',
                                types={'Pheno': hl.tfloat})

        mt = hl.import_vcf(resource('regressionLinear.vcf'))
        mt = hl.linear_regression(y=pheno[mt.s].Pheno,
                                  x=mt.GT.n_alt_alleles())

        results = dict(mt.aggregate_rows(hl.agg.collect((mt.locus.position, mt.linreg))))

        self.assertAlmostEqual(results[1].beta, -0.25, places=6)
        self.assertAlmostEqual(results[1].standard_error, 0.4841229, places=6)
        self.assertAlmostEqual(results[1].t_stat, -0.5163978, places=6)
        self.assertAlmostEqual(results[1].p_value, 0.63281250, places=6)

        self.assertAlmostEqual(results[2].beta, -0.250000, places=6)
        self.assertAlmostEqual(results[2].standard_error, 0.2602082, places=6)
        self.assertAlmostEqual(results[2].t_stat, -0.9607689, places=6)
        self.assertAlmostEqual(results[2].p_value, 0.391075888, places=6)

        self.assertTrue(np.isnan(results[6].standard_error))
        self.assertTrue(np.isnan(results[7].standard_error))
        self.assertTrue(np.isnan(results[8].standard_error))
        self.assertTrue(np.isnan(results[9].standard_error))
        self.assertTrue(np.isnan(results[10].standard_error))

    def test_linear_regression_with_import_fam_boolean(self):
        covariates = hl.import_table(resource('regressionLinear.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        fam = hl.import_fam(resource('regressionLinear.fam'))
        mt = hl.import_vcf(resource('regressionLinear.vcf'))
        mt = hl.linear_regression(y=fam[mt.s].is_case,
                                  x=mt.GT.n_alt_alleles(),
                                  covariates=list(covariates[mt.s].values()))

        results = dict(mt.aggregate_rows(hl.agg.collect((mt.locus.position, mt.linreg))))

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
        mt = hl.linear_regression(y=fam[mt.s].quant_pheno,
                                  x=mt.GT.n_alt_alleles(),
                                  covariates=list(covariates[mt.s].values()))

        results = dict(mt.aggregate_rows(hl.agg.collect((mt.locus.position, mt.linreg))))

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
        mt = hl.linear_regression(y=pheno[mt.s].Pheno,
                                  x=mt.GT.n_alt_alleles(),
                                  covariates=list(covariates[mt.s].values()),
                                  root='single')
        mt = hl.linear_regression(y=[pheno[mt.s].Pheno, pheno[mt.s].Pheno],
                                  x=mt.GT.n_alt_alleles(),
                                  covariates=list(covariates[mt.s].values()),
                                  root='multi')

        def eq(x1, x2):
            return (hl.is_nan(x1) & hl.is_nan(x2)) | (hl.abs(x1 - x2) < 1e-4)

        self.assertTrue(mt.aggregate_rows(hl.agg.all((eq(mt.single.p_value, mt.multi.p_value[0]) &
                                                      eq(mt.single.standard_error, mt.multi.standard_error[0]) &
                                                      eq(mt.single.t_stat, mt.multi.t_stat[0]) &
                                                      eq(mt.single.beta, mt.multi.beta[0]) &
                                                      eq(mt.single.y_transpose_x, mt.multi.y_transpose_x[0])))))
        self.assertTrue(mt.aggregate_rows(hl.agg.all(eq(mt.multi.p_value[1], mt.multi.p_value[0]) &
                                                     eq(mt.multi.standard_error[1], mt.multi.standard_error[0]) &
                                                     eq(mt.multi.t_stat[1], mt.multi.t_stat[0]) &
                                                     eq(mt.multi.beta[1], mt.multi.beta[0]) &
                                                     eq(mt.multi.y_transpose_x[1], mt.multi.y_transpose_x[0]))))

    def test_logistic_regression_wald_test_two_cov(self):
        covariates = hl.import_table(resource('regressionLogistic.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLogisticBoolean.pheno'),
                                key='Sample',
                                missing='0',
                                types={'isCase': hl.tbool})
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))
        mt = hl.logistic_regression('wald',
                                    y=pheno[mt.s].isCase,
                                    x=mt.GT.n_alt_alleles(),
                                    covariates=[covariates[mt.s].Cov1, covariates[mt.s].Cov2])

        results = dict(mt.aggregate_rows(hl.agg.collect((mt.locus.position, mt.logreg))))

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

        self.assertTrue(is_constant(results[3]))
        self.assertTrue(is_constant(results[6]))
        self.assertTrue(is_constant(results[7]))
        self.assertTrue(is_constant(results[8]))
        self.assertTrue(is_constant(results[9]))
        self.assertTrue(is_constant(results[10]))

    def test_logistic_regression_wald_test_two_cov_pl(self):
        covariates = hl.import_table(resource('regressionLogistic.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLogisticBoolean.pheno'),
                                key='Sample',
                                missing='0',
                                types={'isCase': hl.tbool})
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))
        mt = hl.logistic_regression('wald',
                                    y=pheno[mt.s].isCase,
                                    x=hl.pl_dosage(mt.PL),
                                    covariates=[covariates[mt.s].Cov1, covariates[mt.s].Cov2])

        results = dict(mt.aggregate_rows(hl.agg.collect((mt.locus.position, mt.logreg))))

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

        self.assertFalse(results[3].fit.converged)
        self.assertTrue(is_constant(results[6]))
        self.assertTrue(is_constant(results[7]))
        self.assertTrue(is_constant(results[8]))
        self.assertTrue(is_constant(results[9]))
        self.assertTrue(is_constant(results[10]))

    def test_logistic_regression_wald_two_cov_dosage(self):
        covariates = hl.import_table(resource('regressionLogistic.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLogisticBoolean.pheno'),
                                key='Sample',
                                missing='0',
                                types={'isCase': hl.tbool})
        mt = hl.import_gen(resource('regressionLogistic.gen'),
                           sample_file=resource('regressionLogistic.sample'))
        mt = hl.logistic_regression('wald',
                                    y=pheno[mt.s].isCase,
                                    x=hl.gp_dosage(mt.GP),
                                    covariates=[covariates[mt.s].Cov1, covariates[mt.s].Cov2])

        results = dict(mt.aggregate_rows(hl.agg.collect((mt.locus.position, mt.logreg))))

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

        self.assertFalse(results[3].fit.converged)
        self.assertTrue(is_constant(results[6]))
        self.assertTrue(is_constant(results[7]))
        self.assertTrue(is_constant(results[8]))
        self.assertTrue(is_constant(results[9]))
        self.assertTrue(is_constant(results[10]))

    def test_logistic_regression_lrt_two_cov(self):
        covariates = hl.import_table(resource('regressionLogistic.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLogisticBoolean.pheno'),
                                key='Sample',
                                missing='0',
                                types={'isCase': hl.tbool})
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))
        mt = hl.logistic_regression('lrt',
                                    y=pheno[mt.s].isCase,
                                    x=mt.GT.n_alt_alleles(),
                                    covariates=[covariates[mt.s].Cov1, covariates[mt.s].Cov2])

        results = dict(mt.aggregate_rows(hl.agg.collect((mt.locus.position, mt.logreg))))

        self.assertAlmostEqual(results[1].beta, -0.81226793796, places=6)
        self.assertAlmostEqual(results[1].chi_sq_stat, 0.1503349167, places=6)
        self.assertAlmostEqual(results[1].p_value, 0.6982155052, places=6)

        self.assertAlmostEqual(results[2].beta, -0.43659460858, places=6)
        self.assertAlmostEqual(results[2].chi_sq_stat, 0.1813968574, places=6)
        self.assertAlmostEqual(results[2].p_value, 0.6701755415, places=6)

        def is_constant(r):
            return (not r.fit.converged) or np.isnan(r.p_value) or abs(r.p_value - 1) < 1e-4

        self.assertFalse(results[3].fit.converged)
        self.assertTrue(is_constant(results[6]))
        self.assertTrue(is_constant(results[7]))
        self.assertTrue(is_constant(results[8]))
        self.assertTrue(is_constant(results[9]))
        self.assertTrue(is_constant(results[10]))

    def test_logistic_regression_score_two_cov(self):
        covariates = hl.import_table(resource('regressionLogistic.cov'),
                                     key='Sample',
                                     types={'Cov1': hl.tfloat, 'Cov2': hl.tfloat})
        pheno = hl.import_table(resource('regressionLogisticBoolean.pheno'),
                                key='Sample',
                                missing='0',
                                types={'isCase': hl.tbool})
        mt = hl.import_vcf(resource('regressionLogistic.vcf'))
        mt = hl.logistic_regression('score',
                                    y=pheno[mt.s].isCase,
                                    x=mt.GT.n_alt_alleles(),
                                    covariates=[covariates[mt.s].Cov1, covariates[mt.s].Cov2])

        results = dict(mt.aggregate_rows(hl.agg.collect((mt.locus.position, mt.logreg))))

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

        mt = hl.logistic_regression('wald',
                                    y=mt.is_case,
                                    x=mt.GT.n_alt_alleles(),
                                    covariates=[mt.is_female, mt.PC1, mt.PC2],
                                    root='wald')
        mt = hl.logistic_regression('lrt',
                                    y=mt.is_case,
                                    x=mt.GT.n_alt_alleles(),
                                    covariates=[mt.is_female, mt.PC1, mt.PC2],
                                    root='lrt')
        mt = hl.logistic_regression('score',
                                    y=mt.is_case,
                                    x=mt.GT.n_alt_alleles(),
                                    covariates=[mt.is_female, mt.PC1, mt.PC2],
                                    root='score')
        mt = hl.logistic_regression('firth',
                                    y=mt.is_case,
                                    x=mt.GT.n_alt_alleles(),
                                    covariates=[mt.is_female, mt.PC1, mt.PC2],
                                    root='firth')

        # 2535 samples from 1K Genomes Project
        # Locus("22", 16060511)  # MAC  623
        # Locus("22", 16115878)  # MAC  370
        # Locus("22", 16115882)  # MAC 1207
        # Locus("22", 16117940)  # MAC    7
        # Locus("22", 16117953)  # MAC   21

        mt = mt.select_rows('wald', 'lrt', 'firth', 'score')
        results = dict(mt.aggregate_rows(hl.agg.collect((mt.locus.position, mt.row))))

        self.assertAlmostEqual(results[16060511].wald.beta, -0.097476, places=4)
        self.assertAlmostEqual(results[16060511].wald.standard_error, 0.087478, places=4)
        self.assertAlmostEqual(results[16060511].wald.z_stat, -1.1143, places=4)
        self.assertAlmostEqual(results[16060511].wald.p_value, 0.26516, places=4)
        self.assertAlmostEqual(results[16060511].lrt.p_value, 0.26475, places=4)
        self.assertAlmostEqual(results[16060511].score.p_value, 0.26499, places=4)
        self.assertAlmostEqual(results[16060511].firth.beta, -0.097079, places=4)
        self.assertAlmostEqual(results[16060511].firth.p_value, 0.26593, places=4)


        self.assertAlmostEqual(results[16115878].wald.beta, -0.052632, places=4)
        self.assertAlmostEqual(results[16115878].wald.standard_error, 0.11272, places=4)
        self.assertAlmostEqual(results[16115878].wald.z_stat, -0.46691, places=4)
        self.assertAlmostEqual(results[16115878].wald.p_value, 0.64056, places=4)
        self.assertAlmostEqual(results[16115878].lrt.p_value, 0.64046, places=4)
        self.assertAlmostEqual(results[16115878].score.p_value, 0.64054, places=4)
        self.assertAlmostEqual(results[16115878].firth.beta, -0.052301, places=4)
        self.assertAlmostEqual(results[16115878].firth.p_value, 0.64197, places=4)

        self.assertAlmostEqual(results[16115882].wald.beta, -0.15598, places=4)
        self.assertAlmostEqual(results[16115882].wald.standard_error, 0.079508, places=4)
        self.assertAlmostEqual(results[16115882].wald.z_stat, -1.9619, places=4)
        self.assertAlmostEqual(results[16115882].wald.p_value, 0.049779, places=4)
        self.assertAlmostEqual(results[16115882].lrt.p_value, 0.049675, places=4)
        self.assertAlmostEqual(results[16115882].score.p_value, 0.049675, places=4)
        self.assertAlmostEqual(results[16115882].firth.beta, -0.15567, places=4)
        self.assertAlmostEqual(results[16115882].firth.p_value, 0.04991, places=4)

        self.assertAlmostEqual(results[16117940].wald.beta, -0.88059, places=4)
        self.assertAlmostEqual(results[16117940].wald.standard_error, 0.83769, places=2)
        self.assertAlmostEqual(results[16117940].wald.z_stat, -1.0512, places=2)
        self.assertAlmostEqual(results[16117940].wald.p_value, 0.29316, places=2)
        self.assertAlmostEqual(results[16117940].lrt.p_value, 0.26984, places=4)
        self.assertAlmostEqual(results[16117940].score.p_value, 0.27828, places=4)
        self.assertAlmostEqual(results[16117940].firth.beta, -0.7524, places=4)
        self.assertAlmostEqual(results[16117940].firth.p_value, 0.30731, places=4)

        self.assertAlmostEqual(results[16117953].wald.beta, 0.54921, places=4)
        self.assertAlmostEqual(results[16117953].wald.standard_error, 0.4517, places=3)
        self.assertAlmostEqual(results[16117953].wald.z_stat, 1.2159, places=3)
        self.assertAlmostEqual(results[16117953].wald.p_value, 0.22403, places=3)
        self.assertAlmostEqual(results[16117953].lrt.p_value, 0.21692, places=4)
        self.assertAlmostEqual(results[16117953].score.p_value, 0.21849, places=4)
        self.assertAlmostEqual(results[16117953].firth.beta, 0.5258, places=4)
        self.assertAlmostEqual(results[16117953].firth.p_value, 0.22562, places=4)

    def test_trio_matrix(self):
        """
        This test depends on certain properties of the trio matrix VCF and
        pedigree structure. This test is NOT a valid test if the pedigree
        includes quads: the trio_matrix method will duplicate the parents
        appropriately, but the genotypes_table and samples_table orthogonal
        paths would require another duplication/explode that we haven't written.
        """
        ped = hl.Pedigree.read(resource('triomatrix.fam'))
        ht = hl.import_fam(resource('triomatrix.fam'))

        mt = hl.import_vcf(resource('triomatrix.vcf'))
        mt = mt.annotate_cols(fam=ht[mt.s])

        dads = ht.filter(hl.is_defined(ht.pat_id))
        dads = dads.select(dads.pat_id, is_dad=True).key_by('pat_id')

        moms = ht.filter(hl.is_defined(ht.mat_id))
        moms = moms.select(moms.mat_id, is_mom=True).key_by('mat_id')

        et = (mt.entries()
              .key_by('s')
              .join(dads, how='left')
              .join(moms, how='left'))
        et = et.annotate(is_dad=hl.is_defined(et.is_dad),
                         is_mom=hl.is_defined(et.is_mom))

        et = (et
            .group_by(et.locus, et.alleles, fam=et.fam.fam_id)
            .aggregate(data=hl.agg.collect(hl.struct(
            role=hl.case().when(et.is_dad, 1).when(et.is_mom, 2).default(0),
            g=hl.struct(GT=et.GT, AD=et.AD, DP=et.DP, GQ=et.GQ, PL=et.PL)))))

        et = et.filter(hl.len(et.data) == 3)
        et = et.select('data').explode('data')

        tt = hl.trio_matrix(mt, ped, complete_trios=True).entries().key_by('locus', 'alleles')
        tt = tt.annotate(fam=tt.proband.fam.fam_id,
                         data=[hl.struct(role=0, g=tt.proband_entry.select('GT', 'AD', 'DP', 'GQ', 'PL')),
                               hl.struct(role=1, g=tt.father_entry.select('GT', 'AD', 'DP', 'GQ', 'PL')),
                               hl.struct(role=2, g=tt.mother_entry.select('GT', 'AD', 'DP', 'GQ', 'PL'))])
        tt = tt.select('fam', 'data').explode('data')
        tt = tt.filter(hl.is_defined(tt.data.g)).key_by('locus', 'alleles', 'fam')

        self.assertEqual(et.key.dtype, tt.key.dtype)
        self.assertEqual(et.row.dtype, tt.row.dtype)
        self.assertTrue(et._same(tt))

        # test annotations
        e_cols = (mt.cols()
            .join(dads, how='left')
            .join(moms, how='left'))
        e_cols = e_cols.annotate(is_dad=hl.is_defined(e_cols.is_dad),
                                 is_mom=hl.is_defined(e_cols.is_mom))
        e_cols = (e_cols.group_by(fam=e_cols.fam.fam_id)
            .aggregate(data=hl.agg.collect(hl.struct(role=hl.case()
                                                     .when(e_cols.is_dad, 1).when(e_cols.is_mom, 2).default(0),
                                                     sa=hl.struct(**e_cols.row.select(*mt.col))))))
        e_cols = e_cols.filter(hl.len(e_cols.data) == 3).select('data').explode('data')

        t_cols = hl.trio_matrix(mt, ped, complete_trios=True).cols()
        t_cols = t_cols.annotate(fam=t_cols.proband.fam.fam_id,
                                 data=[
                                     hl.struct(role=0, sa=t_cols.proband),
                                     hl.struct(role=1, sa=t_cols.father),
                                     hl.struct(role=2, sa=t_cols.mother)]).key_by('fam').select('data').explode('data')
        t_cols = t_cols.filter(hl.is_defined(t_cols.data.sa))

        self.assertEqual(e_cols.key.dtype, t_cols.key.dtype)
        self.assertEqual(e_cols.row.dtype, t_cols.row.dtype)
        self.assertTrue(e_cols._same(t_cols))

    def test_sample_qc(self):
        dataset = self.get_dataset()
        dataset = hl.sample_qc(dataset)

    def test_variant_qc(self):
        data = [
            {'v': '1:1:A:T', 's': '1', 'GT': hl.Call([0, 0]), 'GQ': 10, 'DP': 0},
            {'v': '1:1:A:T', 's': '2', 'GT': hl.Call([1, 1]), 'GQ': 10, 'DP': 5},
            {'v': '1:1:A:T', 's': '3', 'GT': hl.Call([0, 1]), 'GQ': 11, 'DP': 100},
            {'v': '1:1:A:T', 's': '4', 'GT': None, 'GQ': None, 'DP': 100},
            {'v': '1:2:A:T,C', 's': '1', 'GT': hl.Call([1, 2]), 'GQ': 10, 'DP': 5},
            {'v': '1:2:A:T,C', 's': '2', 'GT': hl.Call([2, 2]), 'GQ': 10, 'DP': 5},
            {'v': '1:2:A:T,C', 's': '3', 'GT': hl.Call([0, 1]), 'GQ': 10, 'DP': 5},
            {'v': '1:2:A:T,C', 's': '4', 'GT': hl.Call([1, 1]), 'GQ': 10, 'DP': 5},
        ]

        ht = hl.Table.parallelize(data, hl.dtype('struct{v: str, s: str, GT: call, GQ: int, DP: int}'))
        ht = ht.transmute(**hl.parse_variant(ht.v))
        mt = ht.to_matrix_table(['locus', 'alleles'], ['s'], partition_key=['locus'])
        mt = hl.variant_qc(mt, 'vqc')
        r = mt.rows().collect()

        self.assertEqual(r[0].vqc.AF, [0.5, 0.5])
        self.assertEqual(r[0].vqc.AC, [3, 3])
        self.assertEqual(r[0].vqc.AN, 6)
        self.assertEqual(r[0].vqc.homozygote_count, [1, 1])
        self.assertEqual(r[0].vqc.n_called, 3)
        self.assertEqual(r[0].vqc.n_not_called, 1)
        self.assertEqual(r[0].vqc.call_rate, 0.75)
        self.assertEqual(r[0].vqc.n_het, 1)
        self.assertEqual(r[0].vqc.n_non_ref, 2)
        self.assertEqual(r[0].vqc.r_expected_het_freq, 0.6)
        self.assertEqual(r[0].vqc.p_hwe, 0.7)
        self.assertEqual(r[0].vqc.dp_stats.min, 0)
        self.assertEqual(r[0].vqc.dp_stats.max, 100)
        self.assertEqual(r[0].vqc.dp_stats.mean, 51.25)
        self.assertAlmostEqual(r[0].vqc.dp_stats.stdev, 48.782040752719645)
        self.assertEqual(r[0].vqc.gq_stats.min, 10)
        self.assertEqual(r[0].vqc.gq_stats.max, 11)
        self.assertAlmostEqual(r[0].vqc.gq_stats.mean, 10.333333333333334)
        self.assertAlmostEqual(r[0].vqc.gq_stats.stdev, 0.47140452079103168)

        self.assertEqual(r[1].vqc.AF, [0.125, 0.5, 0.375])
        self.assertEqual(r[1].vqc.AC, [1, 4, 3])
        self.assertEqual(r[1].vqc.AN, 8)
        self.assertEqual(r[1].vqc.homozygote_count, [0, 1, 1])
        self.assertEqual(r[1].vqc.n_called, 4)
        self.assertEqual(r[1].vqc.n_not_called, 0)
        self.assertEqual(r[1].vqc.call_rate, 1.0)
        self.assertEqual(r[1].vqc.n_het, 2)
        self.assertEqual(r[1].vqc.n_non_ref, 4)
        self.assertEqual(r[1].vqc.p_hwe, None)
        self.assertEqual(r[1].vqc.r_expected_het_freq, None)
        self.assertEqual(r[1].vqc.dp_stats.min, 5)
        self.assertEqual(r[1].vqc.dp_stats.max, 5)
        self.assertEqual(r[1].vqc.dp_stats.mean, 5)
        self.assertEqual(r[1].vqc.dp_stats.stdev, 0.0)
        self.assertEqual(r[1].vqc.gq_stats.min, 10)
        self.assertEqual(r[1].vqc.gq_stats.max, 10)
        self.assertEqual(r[1].vqc.gq_stats.mean, 10)
        self.assertEqual(r[1].vqc.gq_stats.stdev, 0)

    def test_grm(self):
        tolerance = 0.001

        def load_id_file(path):
            ids = []
            with hl.hadoop_open(path) as f:
                for l in f:
                    r = l.strip().split('\t')
                    self.assertEqual(len(r), 2)
                    ids.append(r[1])
            return ids

        def load_rel(ns, path):
            rel = np.zeros((ns, ns))
            with hl.hadoop_open(path) as f:
                for i, l in enumerate(f):
                    for j, n in enumerate(map(float, l.strip().split('\t'))):
                        rel[i, j] = n
                    self.assertEqual(j, i)
                self.assertEqual(i, ns - 1)
            return rel

        def load_grm(ns, nv, path):
            m = np.zeros((ns, ns))
            with utils.hadoop_open(path) as f:
                i = 0
                for l in f:
                    row = l.strip().split('\t')
                    self.assertEqual(int(row[2]), nv)
                    m[int(row[0]) - 1, int(row[1]) - 1] = float(row[3])
                    i += 1

                self.assertEqual(i, ns * (ns + 1) / 2)
            return m

        def load_bin(ns, path):
            m = np.zeros((ns, ns))
            with utils.hadoop_open(path, 'rb') as f:
                for i in range(ns):
                    for j in range(i + 1):
                        b = f.read(4)
                        self.assertEqual(len(b), 4)
                        m[i, j] = unpack('<f', bytearray(b))[0]
                left = f.read()
                self.assertEqual(len(left), 0)
            return m

        b_file = utils.new_temp_file(prefix="plink")
        rel_file = utils.new_temp_file(prefix="test", suffix="rel")
        rel_id_file = utils.new_temp_file(prefix="test", suffix="rel.id")
        grm_file = utils.new_temp_file(prefix="test", suffix="grm")
        grm_bin_file = utils.new_temp_file(prefix="test", suffix="grm.bin")
        grm_nbin_file = utils.new_temp_file(prefix="test", suffix="grm.N.bin")

        dataset = self.get_dataset()
        n_samples = dataset.count_cols()
        dataset = dataset.annotate_rows(AC=agg.sum(dataset.GT.n_alt_alleles()),
                                        n_called=agg.count_where(hl.is_defined(dataset.GT)))
        dataset = dataset.filter_rows((dataset.AC > 0) & (dataset.AC < 2 * dataset.n_called))
        dataset = dataset.filter_rows(dataset.n_called == n_samples).persist()

        hl.export_plink(dataset, b_file, ind_id=dataset.s)

        sample_ids = [row.s for row in dataset.cols().select().collect()]
        n_variants = dataset.count_rows()
        self.assertGreater(n_variants, 0)

        grm = hl.genetic_relatedness_matrix(dataset.GT)
        grm.export_id_file(rel_id_file)

        ############
        ### rel

        p_file = utils.new_temp_file(prefix="plink")
        syscall('''plink --bfile {} --make-rel --out {}'''
                .format(utils.uri_path(b_file), utils.uri_path(p_file)), shell=True, stdout=DEVNULL, stderr=DEVNULL)
        self.assertEqual(load_id_file(p_file + ".rel.id"), sample_ids)

        grm.export_rel(rel_file)
        self.assertEqual(load_id_file(rel_id_file), sample_ids)
        self.assertTrue(np.allclose(load_rel(n_samples, p_file + ".rel"),
                                    load_rel(n_samples, rel_file),
                                    atol=tolerance))

        ############
        ### gcta-grm

        p_file = utils.new_temp_file(prefix="plink")
        syscall('''plink --bfile {} --make-grm-gz --out {}'''
                .format(utils.uri_path(b_file), utils.uri_path(p_file)), shell=True, stdout=DEVNULL, stderr=DEVNULL)
        self.assertEqual(load_id_file(p_file + ".grm.id"), sample_ids)

        grm.export_gcta_grm(grm_file)
        self.assertTrue(np.allclose(load_grm(n_samples, n_variants, p_file + ".grm.gz"),
                                    load_grm(n_samples, n_variants, grm_file),
                                    atol=tolerance))

        ############
        ### gcta-grm-bin

        p_file = utils.new_temp_file(prefix="plink")
        syscall('''plink --bfile {} --make-grm-bin --out {}'''
                .format(utils.uri_path(b_file), utils.uri_path(p_file)), shell=True, stdout=DEVNULL, stderr=DEVNULL)

        self.assertEqual(load_id_file(p_file + ".grm.id"), sample_ids)

        grm.export_gcta_grm_bin(grm_bin_file, grm_nbin_file)

        self.assertTrue(np.allclose(load_bin(n_samples, p_file + ".grm.bin"),
                                    load_bin(n_samples, grm_bin_file),
                                    atol=tolerance))
        self.assertTrue(np.allclose(load_bin(n_samples, p_file + ".grm.N.bin"),
                                    load_bin(n_samples, grm_nbin_file),
                                    atol=tolerance))

    def test_block_matrix_from_numpy(self):
        ndarray = np.matrix([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], dtype=np.float64)

        for block_size in [1, 2, 5, 1024]:
            block_matrix = BlockMatrix.from_numpy(ndarray, block_size)
            assert (block_matrix.n_rows == 3)
            assert (block_matrix.n_cols == 5)
            assert (block_matrix.to_numpy() == ndarray).all()

    def test_rrm(self):
        seed = 0
        n1 = 100
        m1 = 200
        k = 3
        fst = .9

        dataset = hl.balding_nichols_model(k,
                                           n1,
                                           m1,
                                           fst=(k * [fst]),
                                           seed=seed,
                                           n_partitions=4)
        dataset = dataset.annotate_cols(s = hl.str(dataset.sample_idx)).key_cols_by('s')

        def direct_calculation(ds):
            ds = BlockMatrix.from_entry_expr(ds['GT'].n_alt_alleles()).to_numpy()

            # filter out constant rows
            isconst = lambda r: any([all([(gt < c + .01) and (gt > c - .01) for gt in r]) for c in range(3)])
            ds = np.array([row for row in ds if not isconst(row)])

            nvariants, nsamples = ds.shape
            sumgt = lambda r: sum([i for i in r if i >= 0])
            sumsq = lambda r: sum([i ** 2 for i in r if i >= 0])

            mean = [sumgt(row) / nsamples for row in ds]
            stddev = [sqrt(sumsq(row) / nsamples - mean[i] ** 2)
                      for i, row in enumerate(ds)]

            mat = np.array([[(g - mean[i]) / stddev[i] for g in row] for i, row in enumerate(ds)])

            rrm = (mat.T @ mat) / nvariants

            return rrm

        def hail_calculation(ds):
            rrm = hl.realized_relationship_matrix(ds.GT)
            fn = utils.new_temp_file(suffix='.tsv')

            rrm.export_tsv(fn)
            data = []
            with open(utils.uri_path(fn)) as f:
                f.readline()
                for line in f:
                    row = line.strip().split()
                    data.append(list(map(float, row)))

            return np.array(data)

        manual = direct_calculation(dataset)
        rrm = hail_calculation(dataset)

        self.assertTrue(np.allclose(manual, rrm))

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
        mt = mt.annotate_rows(AC = hl.agg.sum(mt.GT.n_alt_alleles()),
                              n_called = hl.agg.count_where(hl.is_defined(mt.GT)))
        mt = mt.filter_rows((mt.AC > 0) & (mt.AC < 2 * mt.n_called)).persist()
        n_rows = mt.count_rows()

        def make_expr(mean):
            return hl.cond(hl.is_defined(mt.GT),
                           (mt.GT.n_alt_alleles() - mean) / hl.sqrt(mean * (2 - mean) * n_rows / 2),
                           0)
        eigen, scores, loadings= hl.pca(hl.bind(make_expr, mt.AC / mt.n_called), k=3, compute_loadings=True)
        hail_scores = scores.explode('scores').scores.collect()
        hail_loadings = loadings.explode('loadings').loadings.collect()

        self.assertEqual(len(eigen), 3)
        self.assertEqual(scores.count(), mt.count_cols())
        self.assertEqual(loadings.count(), n_rows)

        # compute PCA with numpy
        def normalize(a):
            ms = np.mean(a, axis = 0, keepdims = True)
            return np.divide(np.subtract(a, ms), np.sqrt(2.0*np.multiply(ms/2.0, 1-ms/2.0)*a.shape[1]))

        g = np.pad(np.diag([1.0, 1, 2]), ((0, 1), (0, 0)), mode='constant')
        g[1, 0] = 1.0 / 3
        n = normalize(g)
        U, s, V = np.linalg.svd(n, full_matrices=0)
        np_scores = U.dot(np.diag(s)).flatten()
        np_loadings = V.transpose().flatten()
        np_eigenvalues = np.multiply(s,s).flatten()


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
        self.assertTrue(rkin.select("ibd0")._same(hkin.select("ibd0"), tolerance=1e-2, absolute=True))
        self.assertTrue(rkin.select("ibd1")._same(hkin.select("ibd1"), tolerance=2e-2, absolute=True))
        self.assertTrue(rkin.select("ibd2")._same(hkin.select("ibd2"), tolerance=1e-2, absolute=True))

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

    def test_rename_duplicates(self):
        dataset = self.get_dataset()  # FIXME - want to rename samples with same id
        renamed_ids = hl.rename_duplicates(dataset).cols().select().collect()
        self.assertTrue(len(set(renamed_ids)), len(renamed_ids))

    def test_split_multi_hts(self):
        ds1 = hl.import_vcf(resource('split_test.vcf'))
        ds1 = hl.split_multi_hts(ds1)
        ds2 = hl.import_vcf(resource('split_test_b.vcf'))
        df = ds1.rows()
        self.assertTrue(df.all((df.locus.position == 1180) | df.was_split))
        ds1 = ds1.drop('was_split', 'a_index')
        self.assertTrue(ds1._same(ds2))

    def test_mendel_errors(self):
        mt = hl.import_vcf(resource('mendel.vcf'))
        ped = hl.Pedigree.read(resource('mendel.fam'))

        men, fam, ind, var = hl.mendel_errors(mt['GT'], ped)

        self.assertEqual(men.key.dtype, hl.tstruct(locus=mt.locus.dtype,
                                                   alleles=hl.tarray(hl.tstr),
                                                   s=hl.tstr))
        self.assertEqual(men.row.dtype, hl.tstruct(locus=mt.locus.dtype,
                                                   alleles=hl.tarray(hl.tstr),
                                                   s=hl.tstr,
                                                   fam_id=hl.tstr,
                                                   mendel_code=hl.tint))
        self.assertEqual(fam.key.dtype, hl.tstruct(pat_id=hl.tstr,
                                                   mat_id=hl.tstr))
        self.assertEqual(fam.row.dtype, hl.tstruct(pat_id=hl.tstr,
                                                   mat_id=hl.tstr,
                                                   fam_id=hl.tstr,
                                                   children=hl.tint,
                                                   errors=hl.tint64,
                                                   snp_errors=hl.tint64))
        self.assertEqual(ind.key.dtype, hl.tstruct(s=hl.tstr))
        self.assertEqual(ind.row.dtype, hl.tstruct(s=hl.tstr,
                                                   fam_id=hl.tstr,
                                                   errors=hl.tint64,
                                                   snp_errors=hl.tint64))
        self.assertEqual(var.key.dtype, hl.tstruct(locus=mt.locus.dtype,
                                                   alleles=hl.tarray(hl.tstr)))
        self.assertEqual(var.row.dtype, hl.tstruct(locus=mt.locus.dtype,
                                                   alleles=hl.tarray(hl.tstr),
                                                   errors=hl.tint64))

        self.assertEqual(men.count(), 41)
        self.assertEqual(fam.count(), 2)
        self.assertEqual(ind.count(), 7)
        self.assertEqual(var.count(), mt.count_rows())

        self.assertEqual(set(fam.select('errors', 'snp_errors').collect()),
                         {
                             hl.utils.Struct(pat_id='Dad1', mat_id='Mom1', errors=41, snp_errors=39),
                             hl.utils.Struct(pat_id='Dad2', mat_id='Mom2', errors=0, snp_errors=0)
                         })

        self.assertEqual(set(ind.select('errors', 'snp_errors').collect()),
                         {
                             hl.utils.Struct(s='Son1', errors=23, snp_errors=22),
                             hl.utils.Struct(s='Dtr1', errors=18, snp_errors=17),
                             hl.utils.Struct(s='Dad1', errors=19, snp_errors=18),
                             hl.utils.Struct(s='Mom1', errors=22, snp_errors=21),
                             hl.utils.Struct(s='Dad2', errors=0, snp_errors=0),
                             hl.utils.Struct(s='Mom2', errors=0, snp_errors=0),
                             hl.utils.Struct(s='Son2', errors=0, snp_errors=0)
                         })

        to_keep = hl.set([
            (hl.Locus("1", 1), ['C', 'CT']),
            (hl.Locus("1", 2), ['C', 'T']),
            (hl.Locus("X", 1), ['C', 'T']),
            (hl.Locus("X", 3), ['C', 'T']),
            (hl.Locus("Y", 1), ['C', 'T']),
            (hl.Locus("Y", 3), ['C', 'T'])
        ])
        self.assertEqual(var.filter(to_keep.contains((var.locus, var.alleles)))
                         .order_by('locus')
                         .select('errors').collect(),
                         [
                             hl.utils.Struct(locus=hl.Locus("1", 1), alleles=['C', 'CT'], errors=2),
                             hl.utils.Struct(locus=hl.Locus("1", 2), alleles=['C', 'T'], errors=1),
                             hl.utils.Struct(locus=hl.Locus("X", 1), alleles=['C', 'T'], errors=2),
                             hl.utils.Struct(locus=hl.Locus("X", 3), alleles=['C', 'T'], errors=1),
                             hl.utils.Struct(locus=hl.Locus("Y", 1), alleles=['C', 'T'], errors=1),
                             hl.utils.Struct(locus=hl.Locus("Y", 3), alleles=['C', 'T'], errors=1),
                         ])

        ped2 = hl.Pedigree.read(resource('mendelWithMissingSex.fam'))
        men2, _, _, _ = hl.mendel_errors(mt['GT'], ped2)

        self.assertTrue(men2.filter(men2.s == 'Dtr1')._same(men.filter(men.s == 'Dtr1')))

    def test_export_vcf(self):
        dataset = hl.import_vcf(resource('sample.vcf.bgz'))
        vcf_metadata = hl.get_vcf_metadata(resource('sample.vcf.bgz'))
        hl.export_vcf(dataset, '/tmp/sample.vcf', metadata=vcf_metadata)
        dataset_imported = hl.import_vcf('/tmp/sample.vcf')
        self.assertTrue(dataset._same(dataset_imported))

        metadata_imported = hl.get_vcf_metadata('/tmp/sample.vcf')
        self.assertDictEqual(vcf_metadata, metadata_imported)

    def test_concordance(self):
        dataset = self.get_dataset()
        glob_conc, cols_conc, rows_conc = hl.concordance(dataset, dataset)

        self.assertEqual(sum([sum(glob_conc[i]) for i in range(5)]), dataset.count_rows() * dataset.count_cols())

        counts = dataset.aggregate_entries(hl.Struct(n_het=agg.count(agg.filter(dataset.GT.is_het(), dataset.GT)),
                                                     n_hom_ref=agg.count(agg.filter(dataset.GT.is_hom_ref(), dataset.GT)),
                                                     n_hom_var=agg.count(agg.filter(dataset.GT.is_hom_var(), dataset.GT)),
                                                     nNoCall=agg.count(
                                                         agg.filter(hl.is_missing(dataset.GT), dataset.GT))))

        self.assertEqual(glob_conc[0][0], 0)
        self.assertEqual(glob_conc[1][1], counts.nNoCall)
        self.assertEqual(glob_conc[2][2], counts.n_hom_ref)
        self.assertEqual(glob_conc[3][3], counts.n_het)
        self.assertEqual(glob_conc[4][4], counts.n_hom_var)
        [self.assertEqual(glob_conc[i][j], 0) for i in range(5) for j in range(5) if i != j]

        self.assertTrue(cols_conc.all(hl.sum(hl.flatten(cols_conc.concordance)) == dataset.count_rows()))
        self.assertTrue(rows_conc.all(hl.sum(hl.flatten(rows_conc.concordance)) == dataset.count_cols()))

        cols_conc.write('/tmp/foo.kt', overwrite=True)
        rows_conc.write('/tmp/foo.kt', overwrite=True)

    def test_import_table_force_bgz(self):
        f = utils.new_temp_file(suffix=".bgz")
        t = utils.range_table(10, 5)
        t.export(f)

        f2 = utils.new_temp_file(suffix=".gz")
        utils.run_command(["cp", utils.uri_path(f), utils.uri_path(f2)])
        t2 = hl.import_table(f2, force_bgz=True, impute=True).key_by('idx')
        self.assertTrue(t._same(t2))

    def test_import_locus_intervals(self):
        interval_file = resource('annotinterall.interval_list')
        t = hl.import_locus_intervals(interval_file, reference_genome='GRCh37')
        nint = t.count()

        i = 0
        with open(interval_file) as f:
            for line in f:
                if len(line.strip()) != 0:
                    i += 1
        self.assertEqual(nint, i)
        self.assertEqual(t.interval.dtype.point_type, hl.tlocus('GRCh37'))

        tmp_file = utils.new_temp_file(prefix="test", suffix="interval_list")
        start = t.interval.start
        end = t.interval.end
        (t
         .key_by(interval=hl.locus_interval(start.contig, start.position, end.position, True, True))
         .select()
         .export(tmp_file, header=False))

        t2 = hl.import_locus_intervals(tmp_file)

        self.assertTrue(t.select()._same(t2))

    def test_import_locus_intervals_no_reference_specified(self):
        interval_file = resource('annotinterall.interval_list')
        t = hl.import_locus_intervals(interval_file, reference_genome=None)
        self.assertTrue(t.count() == 2)
        self.assertEqual(t.interval.dtype.point_type, hl.tstruct(contig=hl.tstr, position=hl.tint32))

    def test_import_locus_intervals_badly_defined_intervals(self):
        interval_file = resource('example3.interval_list')
        t = hl.import_locus_intervals(interval_file, reference_genome='GRCh37', skip_invalid_intervals=True)
        self.assertTrue(t.count() == 21)

        t = hl.import_locus_intervals(interval_file, reference_genome=None, skip_invalid_intervals=True)
        self.assertTrue(t.count() == 22)

    def test_import_bed(self):
        bed_file = resource('example1.bed')
        bed = hl.import_bed(bed_file, reference_genome='GRCh37')

        nbed = bed.count()
        i = 0
        with open(bed_file) as f:
            for line in f:
                if len(line.strip()) != 0:
                    try:
                        int(line.split()[0])
                        i += 1
                    except:
                        pass
        self.assertEqual(nbed, i)

        self.assertEqual(bed.interval.dtype.point_type, hl.tlocus('GRCh37'))

        bed_file = resource('example2.bed')
        t = hl.import_bed(bed_file, reference_genome='GRCh37')
        self.assertEqual(t.interval.dtype.point_type, hl.tlocus('GRCh37'))
        self.assertTrue(list(t.key.dtype) == ['interval'])
        self.assertTrue(list(t.row.dtype) == ['interval','target'])

    def test_import_bed_no_reference_specified(self):
        bed_file = resource('example1.bed')
        t = hl.import_bed(bed_file, reference_genome=None)
        self.assertTrue(t.count() == 3)
        self.assertEqual(t.interval.dtype.point_type, hl.tstruct(contig=hl.tstr, position=hl.tint32))

    def test_import_bed_badly_defined_intervals(self):
        bed_file = resource('example4.bed')
        t = hl.import_bed(bed_file, reference_genome='GRCh37', skip_invalid_intervals=True)
        self.assertTrue(t.count() == 3)

        t = hl.import_bed(bed_file, reference_genome=None, skip_invalid_intervals=True)
        self.assertTrue(t.count() == 4)

    def test_annotate_intervals(self):
        ds = self.get_dataset()

        bed1 = hl.import_bed(resource('example1.bed'), reference_genome='GRCh37')
        bed2 = hl.import_bed(resource('example2.bed'), reference_genome='GRCh37')
        bed3 = hl.import_bed(resource('example3.bed'), reference_genome='GRCh37')
        self.assertTrue(list(bed2.key.dtype) == ['interval'])
        self.assertTrue(list(bed2.row.dtype) == ['interval','target'])

        interval_list1 = hl.import_locus_intervals(resource('exampleAnnotation1.interval_list'))
        interval_list2 = hl.import_locus_intervals(resource('exampleAnnotation2.interval_list'))
        self.assertTrue(list(interval_list2.key.dtype) == ['interval'])
        self.assertTrue(list(interval_list2.row.dtype) == ['interval', 'target'])

        ann = ds.annotate_rows(in_interval = bed1[ds.locus]).rows()
        self.assertTrue(ann.all((ann.locus.position <= 14000000) |
                                (ann.locus.position >= 17000000) |
                                (hl.is_missing(ann.in_interval))))

        for bed in [bed2, bed3]:
            ann = ds.annotate_rows(target = bed[ds.locus].target).rows()
            expr = (hl.case()
                    .when(ann.locus.position <= 14000000, ann.target == 'gene1')
                    .when(ann.locus.position >= 17000000, ann.target == 'gene2')
                    .default(ann.target == hl.null(hl.tstr)))
            self.assertTrue(ann.all(expr))

        self.assertTrue(ds.annotate_rows(in_interval = interval_list1[ds.locus]).rows()
                        ._same(ds.annotate_rows(in_interval = bed1[ds.locus]).rows()))

        self.assertTrue(ds.annotate_rows(target = interval_list2[ds.locus].target).rows()
                        ._same(ds.annotate_rows(target = bed2[ds.locus].target).rows()))

    def test_import_fam(self):
        fam_file = resource('sample.fam')
        nfam = hl.import_fam(fam_file).count()
        i = 0
        with open(fam_file) as f:
            for line in f:
                if len(line.strip()) != 0:
                    i += 1
        self.assertEqual(nfam, i)

    def test_export_plink(self):
        vcf_file = resource('sample.vcf')
        mt = hl.split_multi_hts(hl.import_vcf(vcf_file, min_partitions=10))

        split_vcf_file = utils.uri_path(utils.new_temp_file())
        hl_output = utils.uri_path(utils.new_temp_file())
        plink_output = utils.uri_path(utils.new_temp_file())
        merge_output = utils.uri_path(utils.new_temp_file())

        hl.export_vcf(mt, split_vcf_file)
        hl.export_plink(mt, hl_output)

        utils.run_command(["plink", "--vcf", split_vcf_file,
                           "--make-bed", "--out", plink_output,
                           "--const-fid", "--keep-allele-order"])

        data = []
        with open(utils.uri_path(plink_output + ".bim")) as file:
            for line in file:
                row = line.strip().split()
                row[1] = ":".join([row[0], row[3], row[5], row[4]])
                data.append("\t".join(row) + "\n")

        with open(plink_output + ".bim", 'w') as f:
            f.writelines(data)

        utils.run_command(["plink", "--bfile", plink_output,
                           "--bmerge", hl_output, "--merge-mode",
                           "6", "--out", merge_output])

        same = True
        with open(merge_output + ".diff") as f:
            for line in f:
                row = line.strip().split()
                if row != ["SNP", "FID", "IID", "NEW", "OLD"]:
                    same = False
                    break

        self.assertTrue(same)

    def test_export_plink_exprs(self):
        ds = self.get_dataset()
        fam_mapping = {'f0': 'fam_id', 'f1': 'ind_id', 'f2': 'pat_id', 'f3': 'mat_id',
                       'f4': 'is_female', 'f5': 'pheno'}
        bim_mapping = {'f0': 'contig', 'f1': 'varid', 'f2': 'cm_position',
                       'f3': 'position', 'f4': 'a1', 'f5': 'a2'}

        # Test default arguments
        out1 = utils.new_temp_file()
        hl.export_plink(ds, out1)
        fam1 = (hl.import_table(out1 + '.fam', no_header=True, impute=False, missing="")
                .rename(fam_mapping))
        bim1 = (hl.import_table(out1 + '.bim', no_header=True, impute=False)
                .rename(bim_mapping))

        self.assertTrue(fam1.all((fam1.fam_id == "0") & (fam1.pat_id == "0") &
                                 (fam1.mat_id == "0") & (fam1.is_female == "0") &
                                 (fam1.pheno == "NA")))
        self.assertTrue(bim1.all((bim1.varid == bim1.contig + ":" + bim1.position + ":" + bim1.a2 + ":" + bim1.a1) &
                                 (bim1.cm_position == "0.0")))

        # Test non-default FAM arguments
        out2 = utils.new_temp_file()
        hl.export_plink(ds, out2, ind_id=ds.s, fam_id=ds.s, pat_id="nope",
                        mat_id="nada", is_female=True, pheno=False)
        fam2 = (hl.import_table(out2 + '.fam', no_header=True, impute=False, missing="")
                .rename(fam_mapping))

        self.assertTrue(fam2.all((fam2.fam_id == fam2.ind_id) & (fam2.pat_id == "nope") &
                                 (fam2.mat_id == "nada") & (fam2.is_female == "2") &
                                 (fam2.pheno == "1")))

        # Test quantitative phenotype
        out3 = utils.new_temp_file()
        hl.export_plink(ds, out3, ind_id=ds.s, pheno=hl.float64(hl.len(ds.s)))
        fam3 = (hl.import_table(out3 + '.fam', no_header=True, impute=False, missing="")
                .rename(fam_mapping))

        self.assertTrue(fam3.all((fam3.fam_id == "0") & (fam3.pat_id == "0") &
                                 (fam3.mat_id == "0") & (fam3.is_female == "0") &
                                 (fam3.pheno != "0") & (fam3.pheno != "NA")))

        # Test non-default BIM arguments
        out4 = utils.new_temp_file()
        hl.export_plink(ds, out4, varid="hello", cm_position=100)
        bim4 = (hl.import_table(out4 + '.bim', no_header=True, impute=False)
                .rename(bim_mapping))

        self.assertTrue(bim4.all((bim4.varid == "hello") & (bim4.cm_position == "100.0")))

        # Test call expr
        out5 = utils.new_temp_file()
        ds_call = ds.annotate_entries(gt_fake=hl.call(0, 0))
        hl.export_plink(ds_call, out5, call=ds_call.gt_fake)
        ds_all_hom_ref = hl.import_plink(out5 + '.bed', out5 + '.bim', out5 + '.fam')
        nerrors = ds_all_hom_ref.aggregate_entries(agg.count_where(~ds_all_hom_ref.GT.is_hom_ref()))
        self.assertTrue(nerrors == 0)

        # Test white-space in FAM id expr raises error
        with self.assertRaisesRegex(TypeError, "has spaces in the following values:"):
            hl.export_plink(ds, utils.new_temp_file(), mat_id="hello world")

        # Test white-space in varid expr raises error
        with self.assertRaisesRegex(utils.FatalError, "no white space allowed:"):
            hl.export_plink(ds, utils.new_temp_file(), varid="hello world")

    def test_export_gen(self):
        gen = hl.import_gen(resource('example.gen'),
                            sample_file=resource('example.sample'),
                            contig_recoding={"01": "1"},
                            reference_genome='GRCh37',
                            min_partitions=3)

        file = '/tmp/test_export_gen'
        hl.export_gen(gen, file)
        gen2 = hl.import_gen(file + '.gen',
                             sample_file=file + '.sample',
                             reference_genome='GRCh37',
                             min_partitions=3)

        self.assertTrue(gen._same(gen2, tolerance=3E-4, absolute=True))

    def test_export_gen_exprs(self):
        gen = hl.import_gen(resource('example.gen'),
                            sample_file=resource('example.sample'),
                            contig_recoding={"01": "1"},
                            reference_genome='GRCh37',
                            min_partitions=3).add_col_index().add_row_index()

        out1 = utils.new_temp_file()
        hl.export_gen(gen, out1, id1=hl.str(gen.col_idx), id2=hl.str(gen.col_idx), missing=0.5,
                      varid=hl.str(gen.row_idx), rsid=hl.str(gen.row_idx), gp=[0.0, 1.0, 0.0])

        in1 = (hl.import_gen(out1 + '.gen', sample_file=out1 + '.sample', min_partitions=3)
               .add_col_index()
               .add_row_index())
        self.assertTrue(in1.aggregate_entries(agg.fraction(in1.GP == [0.0, 1.0, 0.0])) == 1.0)
        self.assertTrue(in1.aggregate_rows(agg.fraction((in1.varid == hl.str(in1.row_idx)) &
                                                        (in1.rsid == hl.str(in1.row_idx)))) == 1.0)
        self.assertTrue(in1.aggregate_cols(agg.fraction((in1.s == hl.str(in1.col_idx)))))

    def test_tdt(self):
        pedigree = hl.Pedigree.read(resource('tdt.fam'))
        tdt_tab = (hl.transmission_disequilibrium_test(
            hl.split_multi_hts(hl.import_vcf(resource('tdt.vcf'), min_partitions=4)),
            pedigree))

        truth = hl.import_table(
            resource('tdt_results.tsv'),
            types={'POSITION': hl.tint32, 'T': hl.tint32, 'U': hl.tint32,
                   'Chi2': hl.tfloat64, 'Pval': hl.tfloat64})
        truth = (truth
            .transmute(locus=hl.locus(truth.CHROM, truth.POSITION),
                       alleles=[truth.REF, truth.ALT])
            .key_by('locus', 'alleles'))

        if tdt_tab.count() != truth.count():
            self.fail('Result has {} rows but should have {} rows'.format(tdt_tab.count(), truth.count()))

        bad = (tdt_tab.filter(hl.is_nan(tdt_tab.p_value), keep=False)
            .join(truth.filter(hl.is_nan(truth.Pval), keep=False), how='outer'))
        bad.describe()

        bad = bad.filter(~(
                (bad.t == bad.T) &
                (bad.u == bad.U) &
                (hl.abs(bad.chi2 - bad.Chi2) < 0.001) &
                (hl.abs(bad.p_value - bad.Pval) < 0.001)))

        if bad.count() != 0:
            bad.order_by(hl.asc(bad.v)).show()
            self.fail('Found rows in violation of the predicate (see show output)')

    def test_maximal_independent_set(self):
        # prefer to remove nodes with higher index
        t = hl.utils.range_table(10)
        graph = t.select(i=hl.int64(t.idx), j=hl.int64(t.idx + 10), bad_type=hl.float32(t.idx))

        mis_table = hl.maximal_independent_set(graph.i, graph.j, True, lambda l, r: l - r)
        mis = [row['node'] for row in mis_table.collect()]
        self.assertEqual(sorted(mis), list(range(0, 10)))
        self.assertEqual(mis_table.row.dtype, hl.tstruct(node=hl.tint64))
        self.assertEqual(mis_table.key.dtype, hl.tstruct(node=hl.tint64))

        self.assertRaises(ValueError, lambda: hl.maximal_independent_set(graph.i, graph.bad_type, True))
        self.assertRaises(ValueError, lambda: hl.maximal_independent_set(graph.i, hl.utils.range_table(10).idx, True))
        self.assertRaises(ValueError, lambda: hl.maximal_independent_set(hl.literal(1), hl.literal(2), True))

    def test_maximal_independent_set2(self):
        edges = [(0, 4), (0, 1), (0, 2), (1, 5), (1, 3), (2, 3), (2, 6),
                 (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
        edges = [{"i": l, "j": r} for l, r in edges]

        t = hl.Table.parallelize(edges, hl.tstruct(i=hl.tint64, j=hl.tint64))
        mis_t = hl.maximal_independent_set(t.i, t.j)
        self.assertTrue(mis_t.row.dtype == hl.tstruct(node=hl.tint64) and
                        mis_t.globals.dtype == hl.tstruct())

        mis = set([row.node for row in mis_t.collect()])
        maximal_indep_sets = [{0, 6, 5, 3}, {1, 4, 7, 2}]
        non_maximal_indep_sets = [{0, 7}, {6, 1}]
        self.assertTrue(mis in non_maximal_indep_sets or mis in maximal_indep_sets)

    def test_maximal_independent_set3(self):
        is_case = {"A", "C", "E", "G", "H"}
        edges = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]
        edges = [{"i": {"id": l, "is_case": l in is_case},
                  "j": {"id": r, "is_case": r in is_case}} for l, r in edges]

        t = hl.Table.parallelize(edges, hl.tstruct(i=hl.tstruct(id=hl.tstr, is_case=hl.tbool),
                                                   j=hl.tstruct(id=hl.tstr, is_case=hl.tbool)))

        tiebreaker = lambda l, r: (hl.case()
                                   .when(l.is_case & (~r.is_case), -1)
                                   .when(~(l.is_case) & r.is_case, 1)
                                   .default(0))

        mis = hl.maximal_independent_set(t.i, t.j, tie_breaker=tiebreaker)

        expected_sets = [{"A", "C", "E", "G"}, {"A", "C", "E", "H"}]

        self.assertTrue(mis.all(mis.node.is_case))
        self.assertTrue(set([row.id for row in mis.select(mis.node.id).collect()]) in expected_sets)

    def test_filter_alleles(self):
        # poor man's Gen
        paths = [resource('sample.vcf'),
                 resource('multipleChromosomes.vcf'),
                 resource('sample2.vcf')]
        for path in paths:
            ds = hl.import_vcf(path)
            self.assertEqual(
                hl.filter_alleles(ds, lambda a, i: False).count_rows(), 0)
            self.assertEqual(hl.filter_alleles(ds, lambda a, i: True).count_rows(), ds.count_rows())

    def test_filter_alleles_hts(self):
        # 1 variant: A:T,G
        ds = hl.import_vcf(resource('filter_alleles/input.vcf'))

        self.assertTrue(
            hl.filter_alleles_hts(ds, lambda a, i: a == 'T', subset=True)
                .drop('old_alleles', 'old_locus', 'new_to_old', 'old_to_new')
                ._same(hl.import_vcf(resource('filter_alleles/keep_allele1_subset.vcf'))))

        self.assertTrue(
            hl.filter_alleles_hts(ds, lambda a, i: a == 'G', subset=True)
            .drop('old_alleles', 'old_locus', 'new_to_old', 'old_to_new')
            ._same(hl.import_vcf(resource('filter_alleles/keep_allele2_subset.vcf')))
        )

        self.assertTrue(
            hl.filter_alleles_hts(ds, lambda a, i: a != 'G', subset=False)
                .drop('old_alleles', 'old_locus', 'new_to_old', 'old_to_new')
                ._same(hl.import_vcf(resource('filter_alleles/keep_allele1_downcode.vcf')))
        )

        (hl.filter_alleles_hts(ds, lambda a, i: a == 'G', subset=False)).old_to_new.show()
        self.assertTrue(
            hl.filter_alleles_hts(ds, lambda a, i: a == 'G', subset=False)
                .drop('old_alleles', 'old_locus', 'new_to_old', 'old_to_new')
                ._same(hl.import_vcf(resource('filter_alleles/keep_allele2_downcode.vcf')))
        )

    def test_ld_prune(self):
        ds = hl.split_multi_hts(hl.import_vcf(resource('sample.vcf')))
        pruned_table = hl.ld_prune(ds.GT, r2=0.2, bp_window_size=1000000)

        filtered_ds = (ds.filter_rows(hl.is_defined(pruned_table[(ds.locus, ds.alleles)])))
        filtered_ds = filtered_ds.annotate_rows(stats=agg.stats(filtered_ds.GT.n_alt_alleles()))
        filtered_ds = filtered_ds.annotate_rows(
            mean=filtered_ds.stats.mean, sd_reciprocal=1 / filtered_ds.stats.stdev)

        n_samples = filtered_ds.count_cols()
        normalized_mean_imputed_genotype_expr = (
            hl.cond(hl.is_defined(filtered_ds['GT']),
                    (filtered_ds['GT'].n_alt_alleles() - filtered_ds['mean'])
                    * filtered_ds['sd_reciprocal'] * (1 / hl.sqrt(n_samples)), 0))

        block_matrix = BlockMatrix.from_entry_expr(normalized_mean_imputed_genotype_expr)
        entries = ((block_matrix @ block_matrix.T) ** 2).entries()

        index_table = filtered_ds.add_row_index().rows().key_by('row_idx').select('locus')
        entries = entries.annotate(locus_i=index_table[entries.i].locus, locus_j=index_table[entries.j].locus)

        contig_filter = entries.locus_i.contig == entries.locus_j.contig
        window_filter = (hl.abs(entries.locus_i.position - entries.locus_j.position)) <= 1000000
        identical_filter = entries.i != entries.j

        self.assertEqual(entries.filter(
            (entries['entry'] >= 0.2) & (contig_filter) & (window_filter) & (identical_filter)).count(), 0)

    def test_ld_prune_inputs(self):
        ds = hl.split_multi_hts(hl.import_vcf(resource('sample.vcf')))
        self.assertRaises(ValueError, lambda: hl.ld_prune(ds.GT, r2=0.2, bp_window_size=1000000, memory_per_core=0))

    def test_ld_prune_no_prune(self):
        ds = hl.balding_nichols_model(n_populations=1, n_samples=10, n_variants=100)
        pruned_table = hl.ld_prune(ds.GT, r2=0.1, bp_window_size=0)
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

    def test_entries(self):
        n_rows, n_cols = 5, 3
        rows = [{'i': i, 'j': j, 'entry': float(i + j)} for i in range(n_rows) for j in range(n_cols)]
        schema = hl.tstruct(i=hl.tint32, j=hl.tint32, entry=hl.tfloat64)
        table = hl.Table.parallelize([hl.struct(i=row['i'], j=row['j'], entry=row['entry']) for row in rows], schema)
        table = table.annotate(i=hl.int64(table.i),
                               j=hl.int64(table.j)).key_by('i', 'j')

        ndarray = np.reshape(list(map(lambda row: row['entry'], rows)), (n_rows, n_cols))

        for block_size in [1, 2, 1024]:
            block_matrix = BlockMatrix.from_numpy(ndarray, block_size)
            entries_table = block_matrix.entries()
            self.assertEqual(entries_table.count(), n_cols * n_rows)
            self.assertEqual(len(entries_table.row), 3)
            self.assertTrue(table._same(entries_table))

    def test_filter_intervals(self):
        ds = hl.import_vcf(resource('sample.vcf'), min_partitions=20)

        self.assertEqual(
            hl.filter_intervals(ds, [hl.parse_locus_interval('20:10639222-10644705')]).count_rows(), 3)

        intervals = [hl.parse_locus_interval('20:10639222-10644700'),
                     hl.parse_locus_interval('20:10644700-10644705')]
        self.assertEqual(hl.filter_intervals(ds, intervals).count_rows(), 3)

        intervals = hl.array([hl.parse_locus_interval('20:10639222-10644700'),
                              hl.parse_locus_interval('20:10644700-10644705')])
        self.assertEqual(hl.filter_intervals(ds, intervals).count_rows(), 3)

        intervals = hl.array([hl.parse_locus_interval('20:10639222-10644700').value,
                              hl.parse_locus_interval('20:10644700-10644705')])
        self.assertEqual(hl.filter_intervals(ds, intervals).count_rows(), 3)

        intervals = [hl.parse_locus_interval('[20:10019093-10026348]').value,
                     hl.parse_locus_interval('[20:17705793-17716416]').value]
        self.assertEqual(hl.filter_intervals(ds, intervals).count_rows(), 4)

    def test_filter_intervals_compound_partition_key(self):
        ds = hl.import_vcf(resource('sample.vcf'), min_partitions=20)
        ds = (ds.annotate_rows(variant=hl.struct(locus=ds.locus, alleles=ds.alleles))
              .key_rows_by('locus', 'alleles'))

        intervals = [hl.Interval(hl.Struct(locus=hl.Locus('20', 10639222), alleles=['A', 'T']),
                                 hl.Struct(locus=hl.Locus('20', 10644700), alleles=['A', 'T']))]
        self.assertEqual(hl.filter_intervals(ds, intervals).count_rows(), 3)

    def test_balding_nichols_model(self):
        from hail.stats import TruncatedBetaDist

        ds = hl.balding_nichols_model(2, 20, 25, 3,
                                      pop_dist=[1.0, 2.0],
                                      fst=[.02, .06],
                                      af_dist=TruncatedBetaDist(a=0.01, b=2.0, min=0.05, max=0.95),
                                      seed=1)

        self.assertEqual(ds.count_cols(), 20)
        self.assertEqual(ds.count_rows(), 25)
        self.assertEqual(ds.n_partitions(), 3)

        glob = ds.globals
        self.assertEqual(glob.n_populations.value, 2)
        self.assertEqual(glob.n_samples.value, 20)
        self.assertEqual(glob.n_variants.value, 25)
        self.assertEqual(glob.pop_dist.value, [1, 2])
        self.assertEqual(glob.fst.value, [.02, .06])
        self.assertEqual(glob.seed.value, 1)
        self.assertEqual(glob.ancestral_af_dist.value,
                         hl.Struct(type='TruncatedBetaDist', a=0.01, b=2.0, min=0.05, max=0.95))

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
                covariates=[ds.cov.Cov1, ds.cov.Cov2],
                logistic=False).count()

        hl.skat(key_expr=ds.gene,
                weight_expr=ds.weight,
                y=ds.pheno,
                x=hl.pl_dosage(ds.PL),
                covariates=[ds.cov.Cov1, ds.cov.Cov2],
                logistic=True).count()

    def test_import_gen(self):
        gen = hl.import_gen(resource('example.gen'),
                            sample_file=resource('example.sample'),
                            contig_recoding={"01": "1"},
                            reference_genome = 'GRCh37').rows()
        self.assertTrue(gen.all(gen.locus.contig == "1"))
        self.assertEqual(gen.count(), 199)
        self.assertEqual(gen.locus.dtype, hl.tlocus('GRCh37'))

    def test_import_gen_no_reference_specified(self):
        gen = hl.import_gen(resource('example.gen'),
                            sample_file=resource('example.sample'),
                            reference_genome=None)

        self.assertTrue(gen.locus.dtype == hl.tstruct(contig=hl.tstr, position=hl.tint32))
        self.assertEqual(gen.count_rows(), 199)

    def test_import_bgen(self):
        hl.index_bgen(resource('example.v11.bgen'))

        bgen_rows = hl.import_bgen(resource('example.v11.bgen'),
                                   entry_fields=['GT', 'GP'],
                                   sample_file=resource('example.sample'),
                                   contig_recoding={'01': '1'},
                                   reference_genome='GRCh37').rows()
        self.assertTrue(bgen_rows.all(bgen_rows.locus.contig == '1'))
        self.assertEqual(bgen_rows.count(), 199)

        hl.index_bgen(resource('example.8bits.bgen'))

        bgen = hl.import_bgen(resource('example.8bits.bgen'),
                              entry_fields=['dosage'],
                              contig_recoding={'01': '1'},
                              reference_genome='GRCh37')
        self.assertEqual(bgen.entry.dtype, hl.tstruct(dosage=hl.tfloat64))

        bgen = hl.import_bgen(resource('example.8bits.bgen'),
                              entry_fields=['GT', 'GP'],
                              sample_file=resource('example.sample'),
                              contig_recoding={'01': '1'},
                              reference_genome='GRCh37')
        self.assertEqual(bgen.entry.dtype, hl.tstruct(GT=hl.tcall, GP=hl.tarray(hl.tfloat64)))
        self.assertEqual(bgen.count_rows(), 199)

        hl.index_bgen(resource('example.10bits.bgen'))
        bgen = hl.import_bgen(resource('example.10bits.bgen'),
                              entry_fields=['GT', 'GP', 'dosage'],
                              contig_recoding={'01': '1'},
                              reference_genome='GRCh37')
        self.assertEqual(bgen.entry.dtype, hl.tstruct(GT=hl.tcall, GP=hl.tarray(hl.tfloat64), dosage=hl.tfloat64))
        self.assertEqual(bgen.locus.dtype, hl.tlocus('GRCh37'))

    def test_import_bgen_no_entry_fields(self):
        hl.index_bgen(resource('example.v11.bgen'))

        bgen = hl.import_bgen(resource('example.v11.bgen'),
                              entry_fields=[],
                              sample_file=resource('example.sample'),
                              contig_recoding={'01': '1'},
                              reference_genome='GRCh37')
        bgen._jvds.typecheck()

    def test_import_bgen_no_reference_specified(self):
        bgen = hl.import_bgen(resource('example.10bits.bgen'),
                              entry_fields=['GT', 'GP', 'dosage'],
                              contig_recoding={'01': '1'},
                              reference_genome=None)
        self.assertTrue(bgen.locus.dtype == hl.tstruct(contig=hl.tstr, position=hl.tint32))
        self.assertEqual(bgen.count_rows(), 199)

    def test_import_vcf(self):
        vcf = hl.split_multi_hts(
            hl.import_vcf(resource('sample2.vcf'),
                          reference_genome=hl.get_reference('GRCh38'),
                          contig_recoding={"22": "chr22"}))

        vcf_table = vcf.rows()
        self.assertTrue(vcf_table.all(vcf_table.locus.contig == "chr22"))
        self.assertTrue(vcf.locus.dtype, hl.tlocus('GRCh37'))

    def test_import_vcf_no_reference_specified(self):
        vcf = hl.import_vcf(resource('sample2.vcf'),
                            reference_genome=None)
        self.assertTrue(vcf.locus.dtype == hl.tstruct(contig=hl.tstr, position=hl.tint32))
        self.assertEqual(vcf.count_rows(), 735)

    def test_import_vcf_bad_reference_allele(self):
        vcf = hl.import_vcf(resource('invalid_base.vcf'))
        self.assertEqual(vcf.count_rows(), 1)

    def test_import_vcf_flags_are_defined(self):
        # issue 3277
        t = hl.import_vcf(resource('sample.vcf')).rows()
        self.assertTrue(t.all(hl.is_defined(t.info.NEGATIVE_TRAIN_SITE) &
                              hl.is_defined(t.info.POSITIVE_TRAIN_SITE) &
                              hl.is_defined(t.info.DB) &
                              hl.is_defined(t.info.DS)))

    def test_import_vcf_can_import_float_array_format(self):
        mt = hl.import_vcf(resource('floating_point_array.vcf'))
        self.assertTrue(mt.aggregate_entries(hl.agg.all(mt.numeric_array == [1.5, 2.5])))

    def test_import_vcf_can_import_negative_numbers(self):
        mt = hl.import_vcf(resource('negative_format_fields.vcf'))
        self.assertTrue(mt.aggregate_entries(hl.agg.all(mt.negative_int == -1) &
                                             hl.agg.all(mt.negative_float == -1.5) &
                                             hl.agg.all(mt.negative_int_array == [-1, -2]) &
                                             hl.agg.all(mt.negative_float_array == [-0.5, -1.5])))

    def test_import_vcf_missing_info_field_elements(self):
        mt = hl.import_vcf(resource('missingInfoArray.vcf'), reference_genome='GRCh37', array_elements_required=False)
        mt = mt.select_rows(FOO=mt.info.FOO, BAR=mt.info.BAR)
        expected = hl.Table.parallelize([{'locus': hl.Locus('X', 16050036), 'alleles': ['A', 'C'],
                                          'FOO': [1, None], 'BAR': [2, None, None]},
                                         {'locus': hl.Locus('X', 16061250), 'alleles': ['T', 'A', 'C'],
                                          'FOO': [None, 2, None], 'BAR': [None, 1.0, None]}],
                                        hl.tstruct(locus=hl.tlocus('GRCh37'), alleles=hl.tarray(hl.tstr),
                                                   FOO=hl.tarray(hl.tint), BAR=hl.tarray(hl.tfloat64)),
                                        key=['locus', 'alleles'])
        self.assertTrue(mt.rows()._same(expected))

    def test_import_vcf_missing_format_field_elements(self):
        mt = hl.import_vcf(resource('missingFormatArray.vcf'), reference_genome='GRCh37', array_elements_required=False)
        mt = mt.select_rows().select_entries('AD', 'PL')

        expected = hl.Table.parallelize([{'locus': hl.Locus('X', 16050036), 'alleles': ['A', 'C'], 's': 'C1046::HG02024',
                                          'AD': [None, None], 'PL': [0, None, 180]},
                                         {'locus': hl.Locus('X', 16050036), 'alleles': ['A', 'C'], 's': 'C1046::HG02025',
                                          'AD': [None, 6], 'PL': [70, None]},
                                         {'locus': hl.Locus('X', 16061250), 'alleles': ['T', 'A', 'C'], 's': 'C1046::HG02024',
                                          'AD': [0, 0, None], 'PL': [396, None, None, 33, None, 0]},
                                         {'locus': hl.Locus('X', 16061250), 'alleles': ['T', 'A', 'C'], 's': 'C1046::HG02025',
                                          'AD': [0, 0, 9], 'PL': [None, None, None]}],
                                        hl.tstruct(locus=hl.tlocus('GRCh37'), alleles=hl.tarray(hl.tstr), s=hl.tstr,
                                                   AD=hl.tarray(hl.tint), PL=hl.tarray(hl.tint)),
                                        key=['locus', 'alleles', 's'])

        self.assertTrue(mt.entries()._same(expected))

    def test_export_import_plink_same(self):
        mt = self.get_dataset()
        mt = mt.select_rows(rsid=hl.delimit([mt.locus.contig, hl.str(mt.locus.position), mt.alleles[0], mt.alleles[1]], ':'),
                            cm_position=15.0)
        mt = mt.select_cols(fam_id=hl.null(hl.tstr), pat_id=hl.null(hl.tstr), mat_id=hl.null(hl.tstr),
                            is_female=hl.null(hl.tbool), is_case=hl.null(hl.tbool))
        mt = mt.select_entries('GT')

        bfile = '/tmp/test_import_export_plink'
        hl.export_plink(mt, bfile, ind_id=mt.s, cm_position=mt.cm_position)

        mt_imported = hl.import_plink(bfile + '.bed', bfile + '.bim', bfile + '.fam',
                                      a2_reference=True, reference_genome='GRCh37')
        self.assertTrue(mt._same(mt_imported))
        self.assertTrue(mt.aggregate_rows(hl.agg.all(mt.cm_position == 15.0)))

    def test_import_plink_empty_fam(self):
        mt = self.get_dataset().drop_cols()
        bfile = '/tmp/test_empty_fam'
        hl.export_plink(mt, bfile, ind_id=mt.s)
        with self.assertRaisesRegex(utils.FatalError, "Empty .fam file"):
            hl.import_plink(bfile + '.bed', bfile + '.bim', bfile + '.fam')

    def test_import_plink_empty_bim(self):
        mt = self.get_dataset().drop_rows()
        bfile = '/tmp/test_empty_bim'
        hl.export_plink(mt, bfile, ind_id=mt.s)
        with self.assertRaisesRegex(utils.FatalError, ".bim file does not contain any variants"):
            hl.import_plink(bfile + '.bed', bfile + '.bim', bfile + '.fam')

    def test_import_plink_a1_major(self):
        mt = self.get_dataset()
        bfile = '/tmp/sample_plink'
        hl.export_plink(mt, bfile, ind_id=mt.s)

        def get_data(a2_reference):
            mt_imported = hl.import_plink(bfile + '.bed', bfile + '.bim',
                                          bfile + '.fam', a2_reference=a2_reference)
            return (hl.variant_qc(mt_imported)
                    .rows()
                    .key_by('rsid'))

        a2 = get_data(a2_reference=True)
        a1 = get_data(a2_reference=False)

        j = (a2.annotate(a1_alleles=a1[a2.rsid].alleles, a1_vqc=a1[a2.rsid].variant_qc)
             .rename({'variant_qc': 'a2_vqc', 'alleles': 'a2_alleles'}))

        self.assertTrue(j.all((j.a1_alleles[0] == j.a2_alleles[1]) &
                              (j.a1_alleles[1] == j.a2_alleles[0]) &
                              (j.a1_vqc.n_not_called == j.a2_vqc.n_not_called) &
                              (j.a1_vqc.n_het == j.a2_vqc.n_het) &
                              (j.a1_vqc.homozygote_count[0] == j.a2_vqc.homozygote_count[1]) &
                              (j.a1_vqc.homozygote_count[1] == j.a2_vqc.homozygote_count[0])))

    def test_import_plink_contig_recoding_w_reference(self):
        vcf = hl.split_multi_hts(
            hl.import_vcf(resource('sample2.vcf'),
                          reference_genome=hl.get_reference('GRCh38'),
                          contig_recoding={"22": "chr22"}))

        hl.export_plink(vcf, '/tmp/sample_plink')

        bfile = '/tmp/sample_plink'
        plink = hl.import_plink(
            bfile + '.bed', bfile + '.bim', bfile + '.fam',
            a2_reference=True,
            contig_recoding={'chr22': '22'},
            reference_genome='GRCh37').rows()
        self.assertTrue(plink.all(plink.locus.contig == "22"))
        self.assertEqual(vcf.count_rows(), plink.count())
        self.assertTrue(plink.locus.dtype, hl.tlocus('GRCh37'))

    def test_import_plink_no_reference_specified(self):
        bfile = resource('fastlmmTest')
        plink = hl.import_plink(bfile + '.bed', bfile + '.bim', bfile + '.fam',
                                reference_genome=None)
        self.assertTrue(plink.locus.dtype == hl.tstruct(contig=hl.tstr, position=hl.tint32))

    def test_import_matrix_table(self):
        mt = hl.import_matrix_table(doctest_resource('matrix1.tsv'),
                                    row_fields={'Barcode': hl.tstr, 'Tissue': hl.tstr, 'Days': hl.tfloat32})
        self.assertEqual(mt['Barcode']._indices, mt._row_indices)
        self.assertEqual(mt['Tissue']._indices, mt._row_indices)
        self.assertEqual(mt['Days']._indices, mt._row_indices)
        self.assertEqual(mt['col_id']._indices, mt._col_indices)
        self.assertEqual(mt['row_id']._indices, mt._row_indices)

        mt.count()

        row_fields = {'f0': hl.tstr, 'f1': hl.tstr, 'f2': hl.tfloat32}
        hl.import_matrix_table(doctest_resource('matrix2.tsv'),
                               row_fields=row_fields, row_key=[]).count()
        hl.import_matrix_table(doctest_resource('matrix3.tsv'),
                               row_fields=row_fields,
                               no_header=True).count()
        hl.import_matrix_table(doctest_resource('matrix3.tsv'),
                               row_fields=row_fields,
                               no_header=True,
                               row_key=[]).count()
        self.assertRaises(hl.utils.FatalError,
                     hl.import_matrix_table,
                          doctest_resource('matrix3.tsv'),
                     row_fields=row_fields,
                     no_header=True,
                     row_key=['foo'])

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

    def test_window_by_locus(self):
        mt = hl.utils.range_matrix_table(100, 2, n_partitions=10)
        mt = mt.annotate_rows(locus=hl.locus('1', mt.row_idx + 1))
        mt = mt.key_rows_by('locus')
        mt = mt.annotate_entries(e_row_idx = mt.row_idx, e_col_idx = mt.col_idx)
        mt = hl.window_by_locus(mt, 5).cache()

        self.assertEqual(mt.count_rows(), 100)

        rows = mt.rows()
        self.assertTrue(rows.all((rows.row_idx < 5) | (rows.prev_rows.length() == 5)))
        self.assertTrue(rows.all(hl.all(lambda x: (rows.row_idx - 1 - x[0]) == x[1].row_idx,
                                        hl.zip_with_index(rows.prev_rows))))

        entries = mt.entries()
        self.assertTrue(entries.all(hl.all(lambda x: x.e_col_idx == entries.col_idx, entries.prev_entries)))
        self.assertTrue(entries.all(hl.all(lambda x: entries.row_idx - 1 - x[0] == x[1].e_row_idx,
                                           hl.zip_with_index(entries.prev_entries))))
