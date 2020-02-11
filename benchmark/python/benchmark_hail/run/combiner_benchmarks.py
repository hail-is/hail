import logging

import hail as hl
import hail.experimental.vcf_combiner as comb

from .resources import empty_gvcf
from .utils import benchmark

COMBINE_GVCF_MAX = 100
MAX_TO_COMBINE = 100 * COMBINE_GVCF_MAX


def chunks(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


@benchmark(args=empty_gvcf.handle())
def compile_10k_merge(path):
    interval = [hl.eval(hl.parse_locus_interval('chr1:START-END', reference_genome='GRCh38'))]
    vcfs = hl.import_vcfs([path], interval, reference_genome='GRCh38')
    vcfs = vcfs * MAX_TO_COMBINE
    mts = [comb.transform_gvcf(vcf) for vcf in vcfs]
    combined = [comb.combine_gvcfs(mts) for mts in chunks(mts, COMBINE_GVCF_MAX)]
    hl.experimental.write_matrix_tables(combined, '/tmp/combiner-temporary/', overwrite=True)
