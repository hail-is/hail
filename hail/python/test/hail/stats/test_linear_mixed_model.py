import unittest
import numpy as np

import hail as hl
import hail.utils as utils
from hail.stats import LinearMixedModel
from hail.linalg import BlockMatrix
from ..helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):

    @staticmethod
    def _filter_and_standardize_cols(a):
        a = a.copy()
        col_means = np.mean(a, axis=0, keepdims=True)
        a -= col_means
        col_lengths = np.linalg.norm(a, axis=0, keepdims=True)
        col_filter = col_lengths > 0
        return np.copy(a[:, np.squeeze(col_filter)] / col_lengths[col_filter])

    @skip_unless_spark_backend()
    def test_linear_mixed_model_fastlmm(self):
        # FastLMM Test data is from all.bed, all.bim, all.fam, cov.txt, pheno_10_causals.txt:
        #   https://github.com/MicrosoftGenomics/FaST-LMM/tree/master/tests/datasets/synth
        #
        # Data is filtered to chromosome 1,3 and samples 0-124,375-499 (2000 variants and 250 samples)
        #
        # Results are computed with single_snp (with LOCO) as in:
        #   https://github.com/MicrosoftGenomics/FaST-LMM/blob/master/doc/ipynb/FaST-LMM.ipynb

        n, m = 250, 1000  # per chromosome

        x_table = hl.import_table(resource('fastlmmCov.txt'), no_header=True, impute=True).key_by('f1')
        y_table = hl.import_table(resource('fastlmmPheno.txt'), no_header=True, impute=True, delimiter=' ').key_by('f1')

        mt = hl.import_plink(bed=resource('fastlmmTest.bed'),
                             bim=resource('fastlmmTest.bim'),
                             fam=resource('fastlmmTest.fam'),
                             reference_genome=None)
        mt = mt.annotate_cols(x=x_table[mt.col_key].f2)
        mt = mt.annotate_cols(y=y_table[mt.col_key].f2).cache()

        x = np.array([np.ones(n), mt.key_cols_by()['x'].collect()]).T
        y = np.array(mt.key_cols_by()['y'].collect())

        mt_chr1 = mt.filter_rows(mt.locus.contig == '1')
        mt_chr3 = mt.filter_rows(mt.locus.contig == '3')

        # testing chrom 1 for h2, betas, p-values
        h2_fastlmm = 0.14276125
        beta_fastlmm = [0.012202061, 0.037718282, -0.033572693, 0.29171541, -0.045644170]

        # FastLMM p-values do not agree to high precision because FastLMM regresses
        # out x from each SNP first and does an F(1, dof)-test on (beta / se)^2
        # (t-test), whereas Hail does likelihood ratio test.
        # We verify below that Hail's p-values remain fixed going forward.
        # fastlmm = [0.84650294, 0.57865098, 0.59050998, 1.6649473e-06, 0.46892059]
        pval_hail = [0.84543084, 0.57596760, 0.58788517, 1.4057279e-06, 0.46578204]

        gamma_fastlmm = h2_fastlmm / (1 - h2_fastlmm)

        g = BlockMatrix.from_entry_expr(mt_chr1.GT.n_alt_alleles()).to_numpy().T
        g_std = self._filter_and_standardize_cols(g)

        # full rank
        k = (g_std @ g_std.T) * (n / m)
        s, u = np.linalg.eigh(k)
        p = u.T
        model = LinearMixedModel(p @ y, p @ x, s)
        model.fit()

        assert np.isclose(model.h_sq, h2_fastlmm)

        h2_std_error = 0.13770773  # hard coded having checked against plot
        assert np.isclose(model.h_sq_standard_error, h2_std_error, 1e-03)

        h_sq_norm_lkhd = model.h_sq_normalized_lkhd()[1:-1]
        argmax = int(100 * h2_fastlmm)
        assert argmax <= np.argmax(h_sq_norm_lkhd) + 1 <= argmax + 1
        assert np.isclose(np.sum(h_sq_norm_lkhd), 1.0)

        mt3_chr3_5var = mt_chr3.filter_rows(mt_chr3.locus.position < 2005)  # first 5
        a = BlockMatrix.from_entry_expr(mt3_chr3_5var.GT.n_alt_alleles()).to_numpy().T

        # FastLMM standardizes each variant to have mean 0 and variance 1.
        a = self._filter_and_standardize_cols(a) * np.sqrt(n)
        pa = p @ a

        model.fit(log_gamma=np.log(gamma_fastlmm))

        res = model.fit_alternatives_numpy(pa, return_pandas=True)

        assert np.allclose(res['beta'], beta_fastlmm)
        assert np.allclose(res['p_value'], pval_hail)

        pa_t_path = utils.new_temp_file(extension='bm')
        BlockMatrix.from_numpy(pa.T).write(pa_t_path, force_row_major=True)

        res = model.fit_alternatives(pa_t_path).to_pandas()

        assert np.allclose(res['beta'], beta_fastlmm)
        assert np.allclose(res['p_value'], pval_hail)

        # low rank
        ld = g_std.T @ g_std
        sl, v = np.linalg.eigh(ld)
        n_eigenvectors = int(np.sum(sl > 1e-10))
        assert n_eigenvectors < n
        sl = sl[-n_eigenvectors:]
        v = v[:, -n_eigenvectors:]
        s = sl * (n / m)
        p = (g_std @ (v / np.sqrt(sl))).T
        model = LinearMixedModel(p @ y, p @ x, s, y, x)
        model.fit()

        assert np.isclose(model.h_sq, h2_fastlmm)
        assert np.isclose(model.h_sq_standard_error, h2_std_error)

        model.fit(log_gamma=np.log(gamma_fastlmm))

        pa = p @ a
        res = model.fit_alternatives_numpy(pa, a, return_pandas=True)

        assert np.allclose(res['beta'], beta_fastlmm)
        assert np.allclose(res['p_value'], pval_hail)

        a_t_path = utils.new_temp_file(extension='bm')
        BlockMatrix.from_numpy(a.T).write(a_t_path, force_row_major=True)

        pa_t_path = utils.new_temp_file(extension='bm')
        BlockMatrix.from_numpy(pa.T).write(pa_t_path, force_row_major=True)

        res = model.fit_alternatives(pa_t_path, a_t_path).to_pandas()

        assert np.allclose(res['beta'], beta_fastlmm)
        assert np.allclose(res['p_value'], pval_hail)

        # testing chrom 3 for h2
        h2_fastlmm = 0.36733240

        g = BlockMatrix.from_entry_expr(mt_chr3.GT.n_alt_alleles()).to_numpy().T
        g_std = self._filter_and_standardize_cols(g)

        # full rank
        k = (g_std @ g_std.T) * (n / m)
        s, u = np.linalg.eigh(k)
        p = u.T
        model = LinearMixedModel(p @ y, p @ x, s)
        model.fit()

        assert np.isclose(model.h_sq, h2_fastlmm)

        h2_std_error = 0.17409641  # hard coded having checked against plot
        assert np.isclose(model.h_sq_standard_error, h2_std_error)

        h_sq_norm_lkhd = model.h_sq_normalized_lkhd()[1:-1]
        argmax = int(100 * h2_fastlmm)
        assert argmax <= np.argmax(h_sq_norm_lkhd) + 1 <= argmax + 1
        assert np.isclose(np.sum(h_sq_norm_lkhd), 1.0)

        # low rank
        l = g_std.T @ g_std
        sl, v = np.linalg.eigh(l)
        n_eigenvectors = int(np.sum(sl > 1e-10))
        assert n_eigenvectors < n
        sl = sl[-n_eigenvectors:]
        v = v[:, -n_eigenvectors:]
        s = sl * (n / m)
        p = (g_std @ (v / np.sqrt(sl))).T
        model = LinearMixedModel(p @ y, p @ x, s, y, x)
        model.fit()

        assert np.isclose(model.h_sq, h2_fastlmm)
        assert np.isclose(model.h_sq_standard_error, h2_std_error)

    @skip_unless_spark_backend()
    def test_linear_mixed_model_math(self):
        gamma = 2.0  # testing at fixed value of gamma
        n, f, m = 4, 2, 3
        y = np.array([0.0, 1.0, 8.0, 9.0])
        x = np.array([[1.0, 0.0],
                      [1.0, 2.0],
                      [1.0, 1.0],
                      [1.0, 4.0]])
        z = np.array([[0.0, 0.0, 1.0],
                      [0.0, 1.0, 2.0],
                      [1.0, 2.0, 4.0],
                      [2.0, 4.0, 8.0]])
        k = z @ z.T
        v = k + np.eye(4) / gamma
        v_inv = np.linalg.inv(v)

        beta = np.linalg.solve(x.T @ v_inv @ x, x.T @ v_inv @ y)
        residual = y - x @ beta
        sigma_sq = 1 / (n - f) * (residual @ v_inv @ residual)
        sv = sigma_sq * v
        neg_log_lkhd = 0.5 * (np.linalg.slogdet(sv)[1] + np.linalg.slogdet(x.T @ np.linalg.inv(sv) @ x)[1])  # plus C

        x_star = np.array([1.0, 0.0, 1.0, 0.0])
        a = x_star.reshape(n, 1)
        x1 = np.hstack([a, x])
        beta1 = np.linalg.solve(x1.T @ v_inv @ x1, x1.T @ v_inv @ y)
        residual1 = y - x1 @ beta1
        chi_sq = n * np.log((residual @ v_inv @ residual) / (residual1 @ v_inv @ residual1))

        # test from_kinship, full-rank fit
        model, p = LinearMixedModel.from_kinship(y, x, k)
        s0, u0 = np.linalg.eigh(k)
        s0 = np.flip(s0, axis=0)
        p0 = np.fliplr(u0).T
        self.assertTrue(model._same(LinearMixedModel(p0 @ y, p0 @ x, s0)))

        model.fit(np.log(gamma))
        self.assertTrue(np.allclose(model.beta, beta))
        self.assertAlmostEqual(model.sigma_sq, sigma_sq)
        self.assertAlmostEqual(model.compute_neg_log_reml(np.log(gamma)), neg_log_lkhd)

        # test full-rank alternative
        pa = p @ a
        stats = model.fit_alternatives_numpy(pa).collect()[0]
        self.assertAlmostEqual(stats.beta, beta1[0])
        self.assertAlmostEqual(stats.chi_sq, chi_sq)

        pa_t_path = utils.new_temp_file()
        BlockMatrix.from_numpy(pa.T).write(pa_t_path, force_row_major=True)
        stats = model.fit_alternatives(pa_t_path).collect()[0]
        self.assertAlmostEqual(stats.beta, beta1[0])
        self.assertAlmostEqual(stats.chi_sq, chi_sq)

        # test from_random_effects, low-rank fit
        s0, p0 = s0[:m], p0[:m, :]
        # test BlockMatrix path
        temp_path = utils.new_temp_file()
        model, _ = LinearMixedModel.from_random_effects(y, x, 
                                                        BlockMatrix.from_numpy(z),
                                                        p_path=temp_path,
                                                        complexity_bound=0)
        lmm = LinearMixedModel(p0 @ y, p0 @ x, s0, y, x, p_path=temp_path)
        self.assertTrue(model._same(lmm))
        # test ndarray path
        model, p = LinearMixedModel.from_random_effects(y, x, z)
        lmm = LinearMixedModel(p0 @ y, p0 @ x, s0, y, x)
        self.assertTrue(model._same(lmm))

        model.fit(np.log(gamma))
        self.assertTrue(np.allclose(model.beta, beta))
        self.assertAlmostEqual(model.sigma_sq, sigma_sq)
        self.assertAlmostEqual(model.compute_neg_log_reml(np.log(gamma)), neg_log_lkhd)

        # test low_rank alternative
        pa = p @ a
        stats = model.fit_alternatives_numpy(pa, a).collect()[0]
        self.assertAlmostEqual(stats.beta, beta1[0])
        self.assertAlmostEqual(stats.chi_sq, chi_sq)

        a_t_path = utils.new_temp_file()
        BlockMatrix.from_numpy(a.T).write(a_t_path, force_row_major=True)
        pa_t_path = utils.new_temp_file()
        BlockMatrix.from_numpy(pa.T).write(pa_t_path, force_row_major=True)
        stats = model.fit_alternatives(pa_t_path, a_t_path).collect()[0]
        self.assertAlmostEqual(stats.beta, beta1[0])
        self.assertAlmostEqual(stats.chi_sq, chi_sq)

    @skip_unless_spark_backend()
    def test_linear_mixed_model_function(self):
        n, f, m = 4, 2, 3
        y = np.array([0.0, 1.0, 8.0, 9.0])
        x = np.array([[1.0, 0.0],
                      [1.0, 2.0],
                      [1.0, 1.0],
                      [1.0, 4.0]])
        z = np.array([[0.0, 0.0, 1.0],
                      [0.0, 1.0, 2.0],
                      [1.0, 2.0, 0.0],
                      [2.0, 0.0, 1.0]])

        p_path = utils.new_temp_file()

        def make_call(gt):
            if gt == 0.0:
                return hl.Call([0, 0])
            if gt == 1.0:
                return hl.Call([0, 1])
            if gt == 2.0:
                return hl.Call([1, 1])

        data = [{'v': j, 's': i, 'y': y[i], 'x1': x[i, 1], 'zt': make_call(z[i, j])}
                for i in range(n) for j in range(m)]
        ht = hl.Table.parallelize(data, hl.dtype('struct{v: int32, s: int32, y: float64, x1: float64, zt: tcall}'))
        mt = ht.to_matrix_table(row_key=['v'], col_key=['s'], col_fields=['x1', 'y'])
        colsort = np.argsort(mt.key_cols_by().s.collect()).tolist()
        mt = mt.choose_cols(colsort)

        rrm = hl.realized_relationship_matrix(mt.zt).to_numpy()

        # kinship path agrees with from_kinship
        model, p = hl.linear_mixed_model(mt.y, [1, mt.x1], k=rrm, p_path=p_path, overwrite=True)
        model0, p0 = LinearMixedModel.from_kinship(y, x, rrm, p_path, overwrite=True)
        assert model0._same(model)
        assert np.allclose(p0, p)

        # random effects path with standardize=True agrees with low-rank rrm
        s0, u0 = np.linalg.eigh(rrm)
        s0 = np.flip(s0, axis=0)[:m]
        p0 = np.fliplr(u0).T[:m, :]
        model, p = hl.linear_mixed_model(mt.y, [1, mt.x1], z_t=mt.zt.n_alt_alleles(), p_path=p_path, overwrite=True)
        model0 = LinearMixedModel(p0 @ y, p0 @ x, s0, y, x, p_path=p_path)
        assert model0._same(model)

        # random effects path with standardize=False agrees with from_random_effects
        model0, p0 = LinearMixedModel.from_random_effects(y, x, z, p_path, overwrite=True)
        model, p = hl.linear_mixed_model(mt.y, [1, mt.x1], z_t=mt.zt.n_alt_alleles(), p_path=p_path, overwrite=True, standardize=False)
        assert model0._same(model)
        assert np.allclose(p0, p.to_numpy())

    @skip_unless_spark_backend()
    def test_linear_mixed_regression_full_rank(self):
        x_table = hl.import_table(resource('fastlmmCov.txt'), no_header=True, impute=True).key_by('f1')
        y_table = hl.import_table(resource('fastlmmPheno.txt'), no_header=True, impute=True, delimiter=' ').key_by('f1')

        mt = hl.import_plink(bed=resource('fastlmmTest.bed'),
                             bim=resource('fastlmmTest.bim'),
                             fam=resource('fastlmmTest.fam'),
                             reference_genome=None)
        mt = mt.annotate_cols(x=x_table[mt.col_key].f2)
        mt = mt.annotate_cols(y=y_table[mt.col_key].f2).cache()
        p_path = utils.new_temp_file()

        h2_fastlmm = 0.142761
        h2_places = 6
        beta_fastlmm = [0.012202061, 0.037718282, -0.033572693, 0.29171541, -0.045644170]
        pval_hail = [0.84543084, 0.57596760, 0.58788517, 1.4057279e-06, 0.46578204]

        mt_chr1 = mt.filter_rows(mt.locus.contig == '1')
        model, _ = hl.linear_mixed_model(y=mt_chr1.y, x=[1, mt_chr1.x], z_t=mt_chr1.GT.n_alt_alleles(), p_path=p_path)
        model.fit()
        self.assertAlmostEqual(model.h_sq, h2_fastlmm, places=h2_places)

        mt_chr3 = mt.filter_rows((mt.locus.contig == '3') & (mt.locus.position < 2005))
        mt_chr3 = mt_chr3.annotate_rows(stats=hl.agg.stats(mt_chr3.GT.n_alt_alleles()))
        ht = hl.linear_mixed_regression_rows((mt_chr3.GT.n_alt_alleles() - mt_chr3.stats.mean) / mt_chr3.stats.stdev,
                                             model)
        assert np.allclose(ht.beta.collect(), beta_fastlmm)
        assert np.allclose(ht.p_value.collect(), pval_hail)

    @skip_unless_spark_backend()
    def test_linear_mixed_regression_low_rank(self):
        x_table = hl.import_table(resource('fastlmmCov.txt'), no_header=True, impute=True).key_by('f1')
        y_table = hl.import_table(resource('fastlmmPheno.txt'), no_header=True, impute=True, delimiter=' ').key_by('f1')

        mt = hl.import_plink(bed=resource('fastlmmTest.bed'),
                             bim=resource('fastlmmTest.bim'),
                             fam=resource('fastlmmTest.fam'),
                             reference_genome=None)
        mt = mt.annotate_cols(x=x_table[mt.col_key].f2)
        mt = mt.annotate_cols(y=y_table[mt.col_key].f2).cache()
        p_path = utils.new_temp_file()

        h2_hail = 0.10001626
        beta_hail = [0.0073201542, 0.039969148, -0.036727875, 0.29852363, -0.049212500]
        pval_hail = [0.90685162, 0.54839177, 0.55001054, 9.85247263e-07, 0.42796507]

        mt_chr1 = mt.filter_rows((mt.locus.contig == '1') & (mt.locus.position < 200))
        model, _ = hl.linear_mixed_model(y=mt_chr1.y, x=[1, mt_chr1.x], z_t=mt_chr1.GT.n_alt_alleles(), p_path=p_path)
        model.fit()
        self.assertTrue(model.low_rank)
        self.assertAlmostEqual(model.h_sq, h2_hail)

        mt_chr3 = mt.filter_rows((mt.locus.contig == '3') & (mt.locus.position < 2005))
        mt_chr3 = mt_chr3.annotate_rows(stats=hl.agg.stats(mt_chr3.GT.n_alt_alleles()))
        ht = hl.linear_mixed_regression_rows((mt_chr3.GT.n_alt_alleles() - mt_chr3.stats.mean) / mt_chr3.stats.stdev,
                                             model)
        assert np.allclose(ht.beta.collect(), beta_hail)
        assert np.allclose(ht.p_value.collect(), pval_hail)

    @skip_unless_spark_backend()
    def test_linear_mixed_regression_pass_through(self):
        x_table = hl.import_table(resource('fastlmmCov.txt'), no_header=True, impute=True).key_by('f1')
        y_table = hl.import_table(resource('fastlmmPheno.txt'), no_header=True, impute=True, delimiter=' ').key_by('f1')

        mt = hl.import_plink(bed=resource('fastlmmTest.bed'),
                             bim=resource('fastlmmTest.bim'),
                             fam=resource('fastlmmTest.fam'),
                             reference_genome=None)
        mt = mt.annotate_cols(x=x_table[mt.col_key].f2)
        mt = mt.annotate_cols(y=y_table[mt.col_key].f2).cache()
        p_path = utils.new_temp_file()

        mt_chr1 = mt.filter_rows((mt.locus.contig == '1') & (mt.locus.position < 200))
        model, _ = hl.linear_mixed_model(y=mt_chr1.y, x=[1, mt_chr1.x], z_t=mt_chr1.GT.n_alt_alleles(), p_path=p_path)
        model.fit(log_gamma=0)

        mt_chr3 = mt.filter_rows((mt.locus.contig == '3') & (mt.locus.position < 2005))
        mt_chr3 = mt_chr3.annotate_rows(stats=hl.agg.stats(mt_chr3.GT.n_alt_alleles()), foo=hl.struct(bar=hl.rand_norm(0, 1)))
        ht = hl.linear_mixed_regression_rows((mt_chr3.GT.n_alt_alleles() - mt_chr3.stats.mean) / mt_chr3.stats.stdev,
                                             model, pass_through=['stats', mt_chr3.foo.bar, mt_chr3.cm_position])

        assert mt_chr3.aggregate_rows(hl.agg.all(mt_chr3.foo.bar == ht[mt_chr3.row_key].bar))
