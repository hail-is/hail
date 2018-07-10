import unittest
from math import sqrt
from struct import unpack
from subprocess import DEVNULL, call as syscall

import numpy as np
import pandas as pd

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

        dataset = get_dataset()
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
        dataset = dataset.annotate_cols(s=hl.str(dataset.sample_idx)).key_cols_by('s')

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

    def test_split_multi_hts(self):
        ds1 = hl.import_vcf(resource('split_test.vcf'))
        ds1 = hl.split_multi_hts(ds1)
        ds2 = hl.import_vcf(resource('split_test_b.vcf'))
        df = ds1.rows()
        self.assertTrue(df.all((df.locus.position == 1180) | df.was_split))
        ds1 = ds1.drop('was_split', 'a_index')
        self.assertTrue(ds1._same(ds2))

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

    @staticmethod
    def _standardize_cols(a):
        a = a.copy()
        col_means = np.mean(a, axis=0, keepdims=True)
        a -= col_means
        col_lengths = np.linalg.norm(a, axis=0, keepdims=True)
        col_filter = col_lengths > 0
        return np.copy(a[:, np.squeeze(col_filter)] / col_lengths[col_filter])

    def test_linear_mixed_model_low_rank(self):
        seed = 0
        n_populations = 8
        fst = n_populations * [.9]
        n_samples = 500
        n_variants = 200
        n_orig_markers = 100
        n_culprits = 10
        n_covariates = 3
        sigma_sq = 1
        tau_sq = 1

        from numpy.random import RandomState
        prng = RandomState(seed)

        x = np.hstack((np.ones(shape=(n_samples, 1)),
                       prng.normal(size=(n_samples, n_covariates - 1))))

        mt = hl.balding_nichols_model(n_populations=n_populations,
                                      n_samples=n_samples,
                                      n_variants=n_variants,
                                      fst=fst,
                                      seed=seed)

        pa_t_path = utils.new_temp_file(suffix='bm')
        a_t_path = utils.new_temp_file(suffix='bm')

        BlockMatrix.write_from_entry_expr(mt.GT.n_alt_alleles(), a_t_path)

        a = BlockMatrix.read(a_t_path).T.to_numpy()
        g = a[:,-n_orig_markers:]
        g_std = self._standardize_cols(g)  # filters constant cols

        n_markers = g_std.shape[1]

        k = (g_std @ g_std.T) * n_samples / n_markers

        beta = np.arange(n_covariates)
        beta_stars = np.array([1] * n_culprits)

        y = prng.multivariate_normal(
            np.hstack((a[:, 0:n_culprits], x)) @ np.hstack((beta_stars, beta)),
            sigma_sq * k + tau_sq * np.eye(n_samples))

        # low rank computation of S, P
        l = g_std.T @ g_std
        sl, v = np.linalg.eigh(l)
        n_eigenvectors = int((sl > 1e-10).sum())
        sl = sl[-n_eigenvectors:]
        v = v[:, -n_eigenvectors:]
        s = sl * (n_samples / n_markers)
        p = (g_std @ (v / np.sqrt(sl))).T

        # compare with full rank S, P
        sk0, uk = np.linalg.eigh(k)
        sk = sk0[-n_eigenvectors:]
        pk = uk[:, -n_eigenvectors:].T
        assert np.allclose(sk, s)
        assert np.allclose(np.abs(pk), np.abs(p))

        # build and fit model
        py = p @ y
        px = p @ x
        pa = p @ a

        model = hl.LinearMixedModel(py, px, s, y, x)
        assert model.n == n_samples
        assert model.f == n_covariates
        assert model.r == n_eigenvectors
        assert model.low_rank

        model.fit()

        # check effect sizes tend to be near 1 for first n_marker alternative models
        BlockMatrix.from_numpy(pa).T.write(pa_t_path, force_row_major=True)
        df_lmm = model.fit_alternatives(pa_t_path, a_t_path).to_pandas()

        assert 0.9 < np.mean(df_lmm['beta'][:n_culprits]) < 1.1

        # compare NumPy and Hail LMM per alternative
        df_numpy = model.fit_alternatives_numpy(pa, a).to_pandas()

        na_numpy = df_numpy.isna().any(axis=1)
        na_lmm = df_lmm.isna().any(axis=1)
 
        assert na_numpy.sum() <= 10
        assert na_lmm.sum() <= 10
        assert np.logical_xor(na_numpy, na_lmm).sum() <= 5

        mask = ~(na_numpy | na_lmm)

        lmm_vs_numpy_p_value = np.sort(np.abs(df_lmm['p_value'][mask] - df_numpy['p_value'][mask]))

        assert lmm_vs_numpy_p_value[10] < 1e-12  # 10 least p-values
        assert lmm_vs_numpy_p_value[-1] < 1e-8   # all p-values

        # compare LinearMixedModel and LinearMixedRegression
        y_bc = hl.literal(list(y.reshape(n_samples)))
        cov1_bc = hl.literal(list(x[:, 1]))
        cov2_bc = hl.literal(list(x[:, 2]))

        mt = mt.annotate_cols(y = y_bc[mt.sample_idx],
                              cov1 = cov1_bc[mt.sample_idx],
                              cov2 = cov2_bc[mt.sample_idx])
        mt = mt.annotate_cols(s = hl.str(mt.sample_idx)).key_cols_by('s')

        mt_markers = mt.filter_rows(mt.locus.position > n_variants - n_orig_markers)
        rrm = hl.realized_relationship_matrix(mt_markers.GT)
        mt_lmr = hl.linear_mixed_regression(rrm, mt.y,
                                            mt.GT.n_alt_alleles(),
                                            [mt.cov1, mt.cov2],
                                            n_eigenvectors=n_eigenvectors)

        delta = mt_lmr.lmmreg_global.delta.value
        assert np.isclose(delta, 1 / model.gamma)

        ht_lmr = mt_lmr.rows()
        df_lmr = ht_lmr.select(beta = ht_lmr.lmmreg.beta,
                               sigma_sq = ht_lmr.lmmreg.sigma_g_squared,
                               chi_sq = ht_lmr.lmmreg.chi_sq_stat,
                               p_value = ht_lmr.lmmreg.p_value).to_pandas()

        na_lmr = df_lmr.isna().any(axis=1)

        assert na_lmr.sum() <= 10
        assert np.logical_xor(na_numpy, na_lmm).sum() <= 5
        mask = ~(na_lmm | na_lmr)

        lmm_vs_lmr_p_value = np.sort(np.abs(df_lmm['p_value'][mask] - df_lmr['p_value'][mask]))

        assert lmm_vs_lmr_p_value[10] < 1e-7  # 10 least p-values
        assert lmm_vs_lmr_p_value[-1] < 2e-3  # all p-values

    def test_linear_mixed_model_full_rank(self):
        seed = 0
        n_populations = 8
        fst = n_populations * [.9]
        n_samples = 200
        n_variants = 500
        n_orig_markers = 500
        n_culprits = 20
        n_covariates = 3
        sigma_sq = 1
        tau_sq = 1

        from numpy.random import RandomState
        prng = RandomState(seed)

        x = np.hstack((np.ones(shape=(n_samples, 1)),
                       prng.normal(size=(n_samples, n_covariates - 1))))

        mt = hl.balding_nichols_model(n_populations=n_populations,
                                      n_samples=n_samples,
                                      n_variants=n_variants,
                                      fst=fst,
                                      seed=seed)

        pa_t_path = utils.new_temp_file(suffix='bm')

        a = BlockMatrix.from_entry_expr(mt.GT.n_alt_alleles()).T.to_numpy()
        g = a[:, -n_orig_markers:]
        g_std = self._standardize_cols(g)  # filters constant cols

        n_markers = g_std.shape[1]

        k = (g_std @ g_std.T) * n_samples / n_markers

        beta = np.arange(n_covariates)
        beta_stars = np.array([1] * n_culprits)

        y = prng.multivariate_normal(
            np.hstack((a[:, 0:n_culprits], x)) @ np.hstack((beta_stars, beta)),
            sigma_sq * k + tau_sq * np.eye(n_samples))

        s, u = np.linalg.eigh(k)
        p = u.T

        # build and fit model
        py = p @ y
        px = p @ x
        pa = p @ a

        model = hl.LinearMixedModel(py, px, s)
        assert model.n == n_samples
        assert model.f == n_covariates
        assert model.r == n_samples
        assert (not model.low_rank)

        model.fit()

        # check effect sizes tend to be near 1 for first n_marker alternative models
        BlockMatrix.from_numpy(pa).T.write(pa_t_path, force_row_major=True)
        df_lmm = model.fit_alternatives(pa_t_path).to_pandas()

        assert 0.9 < np.mean(df_lmm['beta'][:n_culprits]) < 1.1

        # compare NumPy and Hail LMM per alternative
        df_numpy = model.fit_alternatives_numpy(pa, a).to_pandas()

        na_numpy = df_numpy.isna().any(axis=1)
        na_lmm = df_lmm.isna().any(axis=1)

        assert na_numpy.sum() <= 20
        assert na_lmm.sum() <= 20
        assert np.logical_xor(na_numpy, na_lmm).sum() <= 10

        mask = ~(na_numpy | na_lmm)

        lmm_vs_numpy_p_value = np.sort(np.abs(df_lmm['p_value'][mask] - df_numpy['p_value'][mask]))

        assert lmm_vs_numpy_p_value[10] < 1e-12  # 10 least p-values
        assert lmm_vs_numpy_p_value[-1] < 1e-8  # all p-values

        # compare LinearMixedModel and LinearMixedRegression
        y_bc = hl.literal(list(y.reshape(n_samples)))
        cov1_bc = hl.literal(list(x[:, 1]))
        cov2_bc = hl.literal(list(x[:, 2]))

        mt = mt.annotate_cols(y = y_bc[mt.sample_idx],
                              cov1 = cov1_bc[mt.sample_idx],
                              cov2 = cov2_bc[mt.sample_idx])
        mt = mt.annotate_cols(s = hl.str(mt.sample_idx)).key_cols_by('s')

        mt_markers = mt.filter_rows(mt.locus.position > n_variants - n_orig_markers)
        rrm = hl.realized_relationship_matrix(mt_markers.GT)
        mt_lmr = hl.linear_mixed_regression(rrm, mt.y,
                                            mt.GT.n_alt_alleles(),
                                            [mt.cov1, mt.cov2])

        delta = mt_lmr.lmmreg_global.delta.value
        assert np.isclose(delta, 1 / model.gamma)

        ht_lmr = mt_lmr.rows()
        df_lmr = ht_lmr.select(beta = ht_lmr.lmmreg.beta,
                               sigma_sq = ht_lmr.lmmreg.sigma_g_squared,
                               chi_sq = ht_lmr.lmmreg.chi_sq_stat,
                               p_value = ht_lmr.lmmreg.p_value).to_pandas()

        na_lmr = df_lmr.isna().any(axis=1)

        assert na_lmr.sum() <= 20
        assert np.logical_xor(na_numpy, na_lmm).sum() <= 10
        mask = ~(na_lmm | na_lmr)

        lmm_vs_lmr_p_value = np.sort(np.abs(df_lmm['p_value'][mask] - df_lmr['p_value'][mask]))

        assert lmm_vs_lmr_p_value[10] < 1e-7  # 10 least p-values
        assert lmm_vs_lmr_p_value[-1] < 2e-3  # all p-values
