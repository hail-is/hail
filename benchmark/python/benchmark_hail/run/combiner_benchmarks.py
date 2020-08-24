import os.path
from tempfile import TemporaryDirectory

import hail as hl

try:
    import hail.experimental.vcf_combiner.vcf_combiner as vc_all
except ImportError:
    vc_all = None

from .resources import empty_gvcf, single_gvcf, chr22_gvcfs
from .utils import benchmark

COMBINE_GVCF_MAX = 100
MAX_TO_COMBINE = 20 * COMBINE_GVCF_MAX


def chunks(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def setup(path):
    interval = [hl.eval(hl.parse_locus_interval('chr1:START-END', reference_genome='GRCh38'))]
    return hl.import_vcfs([path], interval, reference_genome='GRCh38')[0]


@benchmark(args=empty_gvcf.handle())
def compile_2k_merge(path):
    vcf = setup(path)
    vcfs = [vc_all.transform_gvcf(vcf)] * COMBINE_GVCF_MAX
    combined = [vc_all.combine_gvcfs(vcfs)] * 20
    with TemporaryDirectory() as tmpdir:
        hl.experimental.write_matrix_tables(combined, os.path.join(tmpdir, 'combiner-multi-write'), overwrite=True)


@benchmark(args=empty_gvcf.handle())
def python_only_10k_transform(path):
    vcf = setup(path)
    vcfs = [vcf] * 10_000
    _ = [vc_all.transform_gvcf(vcf) for vcf in vcfs]

@benchmark(args=empty_gvcf.handle())
def python_only_10k_combine(path):
    vcf = setup(path)
    mt = vc_all.transform_gvcf(vcf)
    mts = [mt] * 10_000
    _ = [vc_all.combine_gvcfs(mts) for mts in chunks(mts, COMBINE_GVCF_MAX)]

@benchmark(args=single_gvcf.handle())
def import_and_transform_gvcf(path):
    size = vc_all.CombinerConfig.default_exome_interval_size
    intervals = vc_all.calculate_even_genome_partitioning('GRCh38', size)

    [mt] = hl.import_gvcfs([path], intervals, reference_genome='GRCh38')
    mt = vc_all.transform_gvcf(mt)
    mt._force_count()

@benchmark(args=single_gvcf.handle())
def import_gvcf_force_count(path):
    size = vc_all.CombinerConfig.default_exome_interval_size
    intervals = vc_all.calculate_even_genome_partitioning('GRCh38', size)

    [mt] = hl.import_gvcfs([path], intervals, reference_genome='GRCh38')
    mt._force_count_rows()

@benchmark(args=[chr22_gvcfs.handle(name) for name in chr22_gvcfs.samples])
def full_combiner_chr22(*paths):
    with TemporaryDirectory() as tmpdir:
        vc_all.run_combiner(list(paths),
                            out_file=tmpdir,
                            tmp_path='/tmp',
                            branch_factor=16,
                            reference_genome='GRCh38',
                            overwrite=True,
                            use_exome_default_intervals=True)