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

    def _test_linear_mixed_model_low_rank(self):
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
                                      af_dist=hl.rand_unif(0.1, 0.9, seed=seed),
                                      seed=seed)

        pa_t_path = utils.new_temp_file(suffix='bm')
        a_t_path = utils.new_temp_file(suffix='bm')

        BlockMatrix.write_from_entry_expr(mt.GT.n_alt_alleles(), a_t_path)

        a = BlockMatrix.read(a_t_path).T.to_numpy()
        g = a[:, -n_orig_markers:]
        g_std = self._filter_and_standardize_cols(g)

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
        n_eigenvectors = int(np.sum(sl > 1e-10))
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

        model = LinearMixedModel(py, px, s, y, x)
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
        assert np.min(df_numpy['chi_sq']) > 0

        na_numpy = df_numpy.isna().any(axis=1)
        na_lmm = df_lmm.isna().any(axis=1)

        assert na_numpy.sum() <= 10
        assert na_lmm.sum() <= 10
        assert np.logical_xor(na_numpy, na_lmm).sum() <= 5

        mask = ~(na_numpy | na_lmm)

        lmm_vs_numpy_p_value = np.sort(np.abs(df_lmm['p_value'][mask] - df_numpy['p_value'][mask]))

        assert lmm_vs_numpy_p_value[10] < 1e-12  # 10 least p-values differences
        assert lmm_vs_numpy_p_value[-1] < 1e-8   # all p-values

    def _test_linear_mixed_model_full_rank(self):
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
                                      af_dist=hl.rand_unif(0.1, 0.9, seed=seed),
                                      seed=seed)

        pa_t_path = utils.new_temp_file(suffix='bm')

        a = BlockMatrix.from_entry_expr(mt.GT.n_alt_alleles()).T.to_numpy()
        g = a[:, -n_orig_markers:]
        g_std = self._filter_and_standardize_cols(g)

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

        model = LinearMixedModel(py, px, s)
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
        assert np.min(df_numpy['chi_sq']) > 0

        na_numpy = df_numpy.isna().any(axis=1)
        na_lmm = df_lmm.isna().any(axis=1)

        assert na_numpy.sum() <= 20
        assert na_lmm.sum() <= 20
        assert np.logical_xor(na_numpy, na_lmm).sum() <= 10

        mask = ~(na_numpy | na_lmm)

        lmm_vs_numpy_p_value = np.sort(np.abs(df_lmm['p_value'][mask] - df_numpy['p_value'][mask]))

        assert lmm_vs_numpy_p_value[10] < 1e-12  # 10 least p-values differences
        assert lmm_vs_numpy_p_value[-1] < 1e-8  # all p-values

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
        assert np.isclose(model.h_sq_standard_error, h2_std_error)

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

        res = model.fit_alternatives_numpy(pa).to_pandas()

        assert np.allclose(res['beta'], beta_fastlmm)
        assert np.allclose(res['p_value'], pval_hail)

        pa_t_path = utils.new_temp_file(suffix='bm')
        BlockMatrix.from_numpy(pa.T).write(pa_t_path, force_row_major=True)

        res = model.fit_alternatives(pa_t_path).to_pandas()

        assert np.allclose(res['beta'], beta_fastlmm)
        assert np.allclose(res['p_value'], pval_hail)

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

        model.fit(log_gamma=np.log(gamma_fastlmm))

        pa = p @ a
        res = model.fit_alternatives_numpy(pa, a).to_pandas()

        assert np.allclose(res['beta'], beta_fastlmm)
        assert np.allclose(res['p_value'], pval_hail)

        a_t_path = utils.new_temp_file(suffix='bm')
        BlockMatrix.from_numpy(a.T).write(a_t_path, force_row_major=True)

        pa_t_path = utils.new_temp_file(suffix='bm')
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

    def test_linear_mixed_model_math(self):
        gamma = 2.0  # testing at fixed value of gamma
        n, p, m = 4, 2, 3
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
        sigma_sq = 1 / (n - p) * (residual @ v_inv @ residual)
        sv = sigma_sq * v
        neg_log_lkhd = 0.5 * (np.linalg.slogdet(sv)[1] + np.linalg.slogdet(x.T @ np.linalg.inv(sv) @ x)[1])  # plus C

        x_star = np.array([1.0, 0.0, 1.0, 0.0])
        a = x_star.reshape(n, 1)
        x1 = np.hstack([a, x])
        beta1 = np.linalg.solve(x1.T @ v_inv @ x1, x1.T @ v_inv @ y)
        residual1 = y - x1 @ beta1
        chi_sq = n * np.log((residual @ v_inv @ residual) / (residual1 @ v_inv @ residual1))

        # test full-rank fit
        model, p = LinearMixedModel.from_kinship(y, x, k)
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

        # test low-rank fit
        model, p = LinearMixedModel.from_random_effects(y, x, z)
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
