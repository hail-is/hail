import os.path
from tempfile import TemporaryDirectory

import hail as hl
import hail.experimental.vcf_combiner as comb

from .resources import empty_gvcf
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
    vcfs = [comb.transform_gvcf(vcf)] * COMBINE_GVCF_MAX
    combined = [comb.combine_gvcfs(vcfs)] * 20
    with TemporaryDirectory() as tmpdir:
        hl.experimental.write_matrix_tables(combined, os.path.join(tmpdir, 'combiner-multi-write'), overwrite=True)


@benchmark(args=empty_gvcf.handle())
def python_only_10k_transform(path):
    vcf = setup(path)
    vcfs = [vcf] * 10_000
    _ = [comb.transform_gvcf(vcf) for vcf in vcfs]

@benchmark(args=empty_gvcf.handle())
def python_only_10k_combine(path):
    vcf = setup(path)
    mt = comb.transform_gvcf(vcf)
    mts = [mt] * 10_000
    _ = [comb.combine_gvcfs(mts) for mts in chunks(mts, COMBINE_GVCF_MAX)]
