from test.hail.helpers import with_flags

import pytest

import hail as hl
from benchmark.tools import chunk
from hail.vds.combiner import combine_variant_datasets, new_combiner, transform_gvcf

COMBINE_GVCF_MAX = 100
MAX_TO_COMBINE = 20 * COMBINE_GVCF_MAX


def import_vcf(path):
    return hl.import_vcf(str(path), reference_genome='GRCh38', force=True)


@pytest.mark.benchmark()
@pytest.mark.xfail(raises=hl.utils.java.FatalError, reason='??')
@with_flags(no_ir_logging='1')
def benchmark_compile_2k_merge(empty_gvcf, tmp_path):
    vcf = import_vcf(empty_gvcf)
    vcfs = [transform_gvcf(vcf, [])] * COMBINE_GVCF_MAX
    combined = [combine_variant_datasets(vcfs)] * 20
    hl.vds.write_variant_datasets(combined, str(tmp_path / 'combiner-multi-write'), overwrite=True)


@pytest.mark.benchmark(mds=1.1, instances=10, iterations=10, burn_in_iterations=10)
@pytest.mark.xtimeout(270)
def benchmark_python_only_10k_transform(empty_gvcf):
    for vcf in [import_vcf(empty_gvcf)] * 10_000:
        transform_gvcf(vcf, [])


@pytest.mark.benchmark(mds=1.1, instances=10, iterations=10, burn_in_iterations=20)
def benchmark_python_only_10k_combine(empty_gvcf):
    vcf = import_vcf(empty_gvcf)
    mt = transform_gvcf(vcf, [])
    for mts in chunk(COMBINE_GVCF_MAX, [mt] * 10_000):
        combine_variant_datasets(mts)


@pytest.mark.benchmark(mds=1.1, instances=20, iterations=10, burn_in_iterations=10)
def benchmark_import_and_transform_gvcf(single_gvcf):
    mt = import_vcf(single_gvcf)
    vds = transform_gvcf(mt, [])
    vds.reference_data._force_count_rows()
    vds.variant_data._force_count_rows()


@pytest.mark.benchmark(mds=1.2, instances=10, iterations=15, burn_in_iterations=8)
def benchmark_import_gvcf_force_count(single_gvcf):
    mt = import_vcf(single_gvcf)
    mt._force_count_rows()


@pytest.fixture
def tmp_and_output_paths(tmp_path):
    tmp = tmp_path / 'tmp'
    tmp.mkdir()
    output = tmp_path / 'output'
    output.mkdir()
    return (tmp, output)


@pytest.mark.benchmark(mds=1.1, instances=10, iterations=5, burn_in_iterations=10)
@pytest.mark.xtimeout(180)
def benchmark_vds_combiner_chr22(chr22_gvcfs, tmp_and_output_paths):
    parts = hl.eval([hl.parse_locus_interval('chr22:start-end', reference_genome='GRCh38')])
    tmp, output = tmp_and_output_paths
    combiner = new_combiner(
        output_path=str(output),
        intervals=parts,
        temp_path=str(tmp),
        gvcf_paths=[str(path) for path in chr22_gvcfs],
        reference_genome='GRCh38',
        branch_factor=16,
        target_records=10000000,
    )
    combiner.run()
