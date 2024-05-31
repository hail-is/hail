import hail as hl

from .resources import *
from .utils import benchmark


@benchmark(args=profile_25.handle('vcf'))
def import_vcf_write(vcf):
    mt = hl.import_vcf(vcf)
    out = hl.utils.new_temp_file(extension='mt')
    mt.write(out)


@benchmark(args=profile_25.handle('vcf'))
def import_vcf_count_rows(vcf):
    mt = hl.import_vcf(vcf)
    mt.count_rows()


@benchmark(args=profile_25.handle('mt'))
def export_vcf(mt_path):
    mt = hl.read_matrix_table(mt_path)
    out = hl.utils.new_temp_file(extension='vcf.bgz')
    hl.export_vcf(mt, out)


@benchmark(args=profile_25.handle('mt'))
def sample_qc(mt_path):
    hl.sample_qc(hl.read_matrix_table(mt_path)).cols()._force_count()


@benchmark(args=profile_25.handle('mt'))
def variant_qc(mt_path):
    hl.variant_qc(hl.read_matrix_table(mt_path)).rows()._force_count()


@benchmark(args=profile_25.handle('mt'))
def variant_and_sample_qc(mt_path):
    mt = hl.read_matrix_table(mt_path)
    hl.sample_qc(hl.variant_qc(mt))._force_count_rows()


@benchmark(args=profile_25.handle('mt'))
def variant_and_sample_qc_nested_with_filters_2(mt_path):
    mt = hl.read_matrix_table(mt_path)
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.call_rate >= 0.8)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.8)
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.call_rate >= 0.98)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.98)
    mt.count()


@benchmark(args=profile_25.handle('mt'))
def variant_and_sample_qc_nested_with_filters_4(mt_path):
    mt = hl.read_matrix_table(mt_path)
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.call_rate >= 0.8)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.8)
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.call_rate >= 0.98)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.98)
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.call_rate >= 0.99)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.99)
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.call_rate >= 0.999)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.999)
    mt.count()


@benchmark(args=profile_25.handle('mt'))
def variant_and_sample_qc_nested_with_filters_4_counts(mt_path):
    mt = hl.read_matrix_table(mt_path)
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.call_rate >= 0.8)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.8)
    counts1 = mt.count()
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.call_rate >= 0.98)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.98)
    counts2 = mt.count()
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.call_rate >= 0.99)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.99)
    counts3 = mt.count()
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.call_rate >= 0.999)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.999)
    mt.count()


@benchmark(args=profile_25.handle('mt'))
def hwe_normalized_pca(mt_path):
    mt = hl.read_matrix_table(mt_path)
    mt = mt.filter_rows(mt.info.AF[0] > 0.01)
    hl.hwe_normalized_pca(mt.GT)


@benchmark(args=profile_25.handle('mt'))
def hwe_normalized_pca_blanczos_small_data_0_iterations(mt_path):
    mt = hl.read_matrix_table(mt_path)
    mt = mt.filter_rows(mt.info.AF[0] > 0.01)
    hl._hwe_normalized_blanczos(mt.GT, q_iterations=0)


@benchmark(args=profile_25.handle('mt'))
def hwe_normalized_pca_blanczos_small_data_10_iterations(mt_path):
    mt = hl.read_matrix_table(mt_path)
    mt = mt.filter_rows(mt.info.AF[0] > 0.01)
    hl._hwe_normalized_blanczos(mt.GT, q_iterations=10)


@benchmark(args=profile_25.handle('mt'))
def split_multi_hts(mt_path):
    mt = hl.read_matrix_table(mt_path)
    hl.split_multi_hts(mt)._force_count_rows()


@benchmark(args=profile_25.handle('mt'))
def split_multi(mt_path):
    mt = hl.read_matrix_table(mt_path)
    hl.split_multi(mt)._force_count_rows()


@benchmark(args=profile_25.handle('mt'))
def concordance(mt_path):
    mt = hl.read_matrix_table(mt_path)
    mt = mt.filter_rows(mt.alleles.length() == 2)
    _, r, c = hl.methods.qc.concordance(mt, mt, _localize_global_statistics=False)
    r._force_count()
    c._force_count()


@benchmark(args=profile_25.handle('mt'))
def genetics_pipeline(mt_path):
    mt = hl.read_matrix_table(mt_path)
    mt = hl.split_multi_hts(mt)
    mt = hl.variant_qc(mt)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate > 0.95)
    mt = mt.filter_rows(mt.variant_qc.AC[1] > 5)
    mt = mt.filter_entries(hl.case().when(hl.is_indel(mt.alleles[0], mt.alleles[1]), mt.GQ > 20).default(mt.GQ > 10))
    mt.write('/tmp/genetics_pipeline.mt', overwrite=True)


@benchmark(args=profile_25.handle('mt'))
def ld_prune_profile_25(mt_path):
    mt = hl.read_matrix_table(mt_path)
    mt = mt.filter_rows(hl.len(mt.alleles) == 2)
    hl.ld_prune(mt.GT)._force_count()


@benchmark(args=profile_25.handle('mt'))
def pc_relate(mt_path):
    mt = hl.read_matrix_table(mt_path)
    mt = mt.annotate_cols(scores=hl.range(2).map(lambda x: hl.rand_unif(0, 1)))
    rel = hl.pc_relate(mt.GT, 0.05, scores_expr=mt.scores, statistics='kin', min_kinship=0.05)
    rel._force_count()


@benchmark(args=balding_nichols_5k_5k.handle())
def pc_relate_5k_5k(mt_path):
    mt = hl.read_matrix_table(mt_path)
    mt = mt.annotate_cols(scores=hl.range(2).map(lambda x: hl.rand_unif(0, 1)))
    rel = hl.pc_relate(mt.GT, 0.05, scores_expr=mt.scores, statistics='kin', min_kinship=0.05)
    rel._force_count()


@benchmark(args=random_doubles.handle('mt'))
def linear_regression_rows(mt_path):
    mt = hl.read_matrix_table(mt_path)
    num_phenos = 100
    num_covs = 20
    pheno_dict = {f"pheno_{i}": hl.rand_unif(0, 1) for i in range(num_phenos)}
    cov_dict = {f"cov_{i}": hl.rand_unif(0, 1) for i in range(num_covs)}
    mt = mt.annotate_cols(**pheno_dict)
    mt = mt.annotate_cols(**cov_dict)
    res = hl.linear_regression_rows(
        y=[mt[key] for key in pheno_dict.keys()], x=mt.x, covariates=[mt[key] for key in cov_dict.keys()]
    )
    res._force_count()


@benchmark(args=random_doubles.handle('mt'))
def linear_regression_rows_nd(mt_path):
    mt = hl.read_matrix_table(mt_path)
    num_phenos = 100
    num_covs = 20
    pheno_dict = {f"pheno_{i}": hl.rand_unif(0, 1) for i in range(num_phenos)}
    cov_dict = {f"cov_{i}": hl.rand_unif(0, 1) for i in range(num_covs)}
    mt = mt.annotate_cols(**pheno_dict)
    mt = mt.annotate_cols(**cov_dict)
    res = hl._linear_regression_rows_nd(
        y=[mt[key] for key in pheno_dict.keys()], x=mt.x, covariates=[mt[key] for key in cov_dict.keys()]
    )
    res._force_count()


@benchmark(args=random_doubles.handle('mt'))
def logistic_regression_rows_wald(mt_path):
    mt = hl.read_matrix_table(mt_path)
    mt = mt.head(2000)
    num_phenos = 5
    num_covs = 2
    pheno_dict = {f"pheno_{i}": hl.rand_bool(0.5, seed=i) for i in range(num_phenos)}
    cov_dict = {f"cov_{i}": hl.rand_unif(0, 1, seed=i) for i in range(num_covs)}
    mt = mt.annotate_cols(**pheno_dict)
    mt = mt.annotate_cols(**cov_dict)
    res = hl.logistic_regression_rows(
        test='wald', y=[mt[key] for key in pheno_dict.keys()], x=mt.x, covariates=[mt[key] for key in cov_dict.keys()]
    )
    res._force_count()


@benchmark(args=random_doubles.handle('mt'))
def logistic_regression_rows_wald_nd(mt_path):
    mt = hl.read_matrix_table(mt_path)
    mt = mt.head(2000)
    num_phenos = 5
    num_covs = 2
    pheno_dict = {f"pheno_{i}": hl.rand_bool(0.5, seed=i) for i in range(num_phenos)}
    cov_dict = {f"cov_{i}": hl.rand_unif(0, 1, seed=i) for i in range(num_covs)}
    mt = mt.annotate_cols(**pheno_dict)
    mt = mt.annotate_cols(**cov_dict)
    res = hl._logistic_regression_rows_nd(
        test='wald', y=[mt[key] for key in pheno_dict.keys()], x=mt.x, covariates=[mt[key] for key in cov_dict.keys()]
    )
    res._force_count()
