from benchmark.utils import resource, benchmark, get_mt

import hail as hl


@benchmark
def import_vcf_write():
    mt = hl.import_vcf(resource('profile.vcf.bgz'))
    out = hl.utils.new_temp_file(suffix='mt')
    mt.write(out)


@benchmark
def import_vcf_count_rows():
    mt = hl.import_vcf(resource('profile.vcf.bgz'))
    mt.count_rows()


@benchmark
def export_vcf():
    mt = hl.read_matrix_table(resource('profile.mt'))
    out = hl.utils.new_temp_file(suffix='vcf.bgz')
    hl.export_vcf(mt, out)


@benchmark
def sample_qc():
    hl.sample_qc(get_mt()).cols()._force_count()


@benchmark
def variant_qc():
    hl.variant_qc(get_mt()).rows()._force_count()


@benchmark
def hwe_normalized_pca():
    mt = get_mt()
    mt = mt.filter_rows(mt.info.AF[0] > 0.01)
    hl.hwe_normalized_pca(mt.GT)
