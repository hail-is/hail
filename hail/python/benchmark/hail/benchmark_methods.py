import hail as hl
from benchmark.tools import benchmark


@benchmark()
def benchmark_import_vcf_write(profile25_vcf, tmp_path):
    mt = hl.import_vcf(str(profile25_vcf))
    out = str(tmp_path / 'out.mt')
    mt.write(out)


@benchmark()
def benchmark_import_vcf_count_rows(profile25_vcf):
    mt = hl.import_vcf(str(profile25_vcf))
    mt.count_rows()


@benchmark()
def benchmark_export_vcf(profile25_mt, tmp_path):
    mt = hl.read_matrix_table(str(profile25_mt))
    out = str(tmp_path / 'out.vcf.bgz')
    hl.export_vcf(mt, out)


@benchmark()
def benchmark_sample_qc(profile25_mt):
    hl.sample_qc(hl.read_matrix_table(str(profile25_mt))).cols()._force_count()


@benchmark()
def benchmark_variant_qc(profile25_mt):
    hl.variant_qc(hl.read_matrix_table(str(profile25_mt))).rows()._force_count()


@benchmark()
def benchmark_variant_and_sample_qc(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    hl.sample_qc(hl.variant_qc(mt))._force_count_rows()


@benchmark()
def benchmark_variant_and_sample_qc_nested_with_filters_2(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.call_rate >= 0.8)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.8)
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.call_rate >= 0.98)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.98)
    mt.count()


@benchmark()
def benchmark_variant_and_sample_qc_nested_with_filters_4(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
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


@benchmark()
def benchmark_variant_and_sample_qc_nested_with_filters_4_counts(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.call_rate >= 0.8)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.8)
    mt.count()
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.call_rate >= 0.98)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.98)
    mt.count()
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.call_rate >= 0.99)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.99)
    mt.count()
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.call_rate >= 0.999)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.999)
    mt.count()


@benchmark()
def benchmark_hwe_normalized_pca(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt = mt.filter_rows(mt.info.AF[0] > 0.01)
    hl.hwe_normalized_pca(mt.GT)


@benchmark()
def benchmark_hwe_normalized_pca_blanczos_small_data_0_iterations(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt = mt.filter_rows(mt.info.AF[0] > 0.01)
    hl._hwe_normalized_blanczos(mt.GT, q_iterations=0)


@benchmark()
def benchmark_hwe_normalized_pca_blanczos_small_data_10_iterations(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt = mt.filter_rows(mt.info.AF[0] > 0.01)
    hl._hwe_normalized_blanczos(mt.GT, q_iterations=10)


@benchmark()
def benchmark_split_multi_hts(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    hl.split_multi_hts(mt)._force_count_rows()


@benchmark()
def benchmark_split_multi(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    hl.split_multi(mt)._force_count_rows()


@benchmark()
def benchmark_concordance(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt = mt.filter_rows(mt.alleles.length() == 2)
    _, r, c = hl.methods.qc.concordance(mt, mt, _localize_global_statistics=False)
    r._force_count()
    c._force_count()


@benchmark()
def benchmark_genetics_pipeline(profile25_mt, tmp_path):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt = hl.split_multi_hts(mt)
    mt = hl.variant_qc(mt)
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols(mt.sample_qc.call_rate > 0.95)
    mt = mt.filter_rows(mt.variant_qc.AC[1] > 5)
    mt = mt.filter_entries(hl.case().when(hl.is_indel(mt.alleles[0], mt.alleles[1]), mt.GQ > 20).default(mt.GQ > 10))
    mt.write(str(tmp_path / 'genetics_pipeline.mt'))


@benchmark()
def benchmark_ld_prune_profile_25(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt = mt.filter_rows(hl.len(mt.alleles) == 2)
    hl.ld_prune(mt.GT)._force_count()


@benchmark()
def benchmark_pc_relate(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt = mt.annotate_cols(scores=hl.range(2).map(lambda x: hl.rand_unif(0, 1)))
    rel = hl.pc_relate(mt.GT, 0.05, scores_expr=mt.scores, statistics='kin', min_kinship=0.05)
    rel._force_count()


@benchmark()
def benchmark_pc_relate_5k_5k(balding_nichols_5k_5k):
    mt = hl.read_matrix_table(str(balding_nichols_5k_5k))
    mt = mt.annotate_cols(scores=hl.range(2).map(lambda x: hl.rand_unif(0, 1)))
    rel = hl.pc_relate(mt.GT, 0.05, scores_expr=mt.scores, statistics='kin', min_kinship=0.05)
    rel._force_count()


@benchmark()
def benchmark_linear_regression_rows(random_doubles_mt):
    mt = hl.read_matrix_table(str(random_doubles_mt))
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


@benchmark()
def benchmark_linear_regression_rows_nd(random_doubles_mt):
    mt = hl.read_matrix_table(str(random_doubles_mt))
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


@benchmark()
def benchmark_logistic_regression_rows_wald(random_doubles_mt):
    mt = hl.read_matrix_table(str(random_doubles_mt))
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


@benchmark()
def benchmark_logistic_regression_rows_wald_nd(random_doubles_mt):
    mt = hl.read_matrix_table(str(random_doubles_mt))
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
